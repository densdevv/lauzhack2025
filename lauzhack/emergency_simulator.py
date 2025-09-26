#!/usr/bin/env python3
"""
Swiss Weather Emergency Simulator
=================================

Dynamic emergency scenario simulator for the Swiss Weather Intelligence System.
Creates realistic emergency weather conditions with real-time alerts and responses.

Perfect for hackathon presentations and system demonstrations.

Author: EPFL Hackathon Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import threading
from typing import Dict, List, Optional
import os
import json


class SwissWeatherEmergencySimulator:
    """
    Emergency weather scenario simulator for Swiss Weather Intelligence System.
    """
    
    def __init__(self):
        """Initialize the emergency simulator."""
        self.is_running = False
        self.current_scenario = None
        self.simulation_data = []
        self.alert_history = []
        self.scenarios = self._define_emergency_scenarios()
        
    def _define_emergency_scenarios(self) -> Dict:
        """Define various emergency weather scenarios."""
        return {
            'heat_wave': {
                'name': '🔥 Extreme Heat Wave Emergency',
                'description': 'Dangerous heat wave with temperatures exceeding 40°C',
                'duration_hours': 72,
                'parameters': {
                    'temperature': {'base': 42.0, 'variation': 6.0, 'trend': 'increasing'},
                    'humidity': {'base': 25.0, 'variation': 15.0, 'trend': 'decreasing'},
                    'pressure': {'base': 1018.0, 'variation': 8.0, 'trend': 'stable'},
                    'wind_speed': {'base': 2.5, 'variation': 3.0, 'trend': 'low'},
                    'solar_radiation': {'base': 850.0, 'variation': 200.0, 'trend': 'high'}
                },
                'alert_triggers': {
                    'critical_temp': 40.0,
                    'extreme_temp': 43.0,
                    'low_humidity': 20.0,
                    'heat_index_danger': 45.0
                }
            },
            
            'severe_storm': {
                'name': '🌪️ Severe Storm System',
                'description': 'Major storm with destructive winds and heavy precipitation',
                'duration_hours': 18,
                'parameters': {
                    'temperature': {'base': 12.0, 'variation': 8.0, 'trend': 'dropping'},
                    'humidity': {'base': 95.0, 'variation': 10.0, 'trend': 'high'},
                    'pressure': {'base': 965.0, 'variation': 25.0, 'trend': 'dropping'},
                    'wind_speed': {'base': 28.0, 'variation': 15.0, 'trend': 'increasing'},
                    'precipitation': {'base': 15.0, 'variation': 20.0, 'trend': 'heavy'},
                    'solar_radiation': {'base': 50.0, 'variation': 30.0, 'trend': 'low'}
                },
                'alert_triggers': {
                    'low_pressure': 970.0,
                    'critical_pressure': 960.0,
                    'high_wind': 25.0,
                    'extreme_wind': 35.0,
                    'heavy_rain': 10.0,
                    'extreme_rain': 20.0
                }
            },
            
            'flash_flood': {
                'name': '🌊 Flash Flood Emergency',
                'description': 'Intense rainfall causing flash flooding conditions',
                'duration_hours': 12,
                'parameters': {
                    'temperature': {'base': 18.0, 'variation': 5.0, 'trend': 'stable'},
                    'humidity': {'base': 98.0, 'variation': 5.0, 'trend': 'saturated'},
                    'pressure': {'base': 995.0, 'variation': 10.0, 'trend': 'low'},
                    'wind_speed': {'base': 8.0, 'variation': 6.0, 'trend': 'gusty'},
                    'precipitation': {'base': 35.0, 'variation': 25.0, 'trend': 'extreme'},
                    'solar_radiation': {'base': 30.0, 'variation': 20.0, 'trend': 'minimal'}
                },
                'alert_triggers': {
                    'extreme_rain': 25.0,
                    'flash_flood_rain': 40.0,
                    'saturated_humidity': 95.0,
                    'rapid_accumulation': 60.0  # mm in 3 hours
                }
            }
        }
    
    def generate_mock_data_point(self, scenario: Dict, hour: int) -> Dict:
        """Generate a single mock data point for the scenario."""
        data_point = {
            'time': datetime.now() + timedelta(hours=hour),
            'scenario': scenario['name'],
            'hour': hour
        }
        
        for param, config in scenario['parameters'].items():
            base_value = config['base']
            variation = config['variation']
            trend = config.get('trend', 'stable')
            
            # Apply trend modifications
            trend_factor = self._apply_trend(trend, hour, scenario['duration_hours'])
            
            # Add realistic noise and variation
            noise = np.random.normal(0, variation * 0.3)
            cycle_variation = variation * 0.5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            
            # Calculate final value
            value = base_value + trend_factor + noise + cycle_variation
            
            # Apply realistic constraints
            value = self._apply_constraints(param, value)
            
            data_point[param] = round(value, 2)
        
        return data_point
    
    def _apply_trend(self, trend: str, hour: int, total_hours: int) -> float:
        """Apply trend modifications based on scenario progression."""
        progress = hour / total_hours
        
        trend_patterns = {
            'increasing': 5.0 * progress,
            'decreasing': -5.0 * progress,
            'dropping': -10.0 * progress,
            'extreme_cold': -5.0 * (1 + progress),
            'foehn_warm': 3.0 * np.sin(np.pi * progress),
            'foehn_gusts': 10.0 * np.sin(2 * np.pi * progress),
            'heavy': 8.0 * np.sin(np.pi * progress),
            'extreme': 15.0 * np.sin(np.pi * progress * 1.5),
            'wind_chill': 8.0 * progress,
            'stable': 0.0,
            'high': 2.0,
            'low': -2.0,
            'minimal': -1.0,
            'saturated': 1.0,
            'extremely_dry': -3.0 * progress,
            'gradient': 5.0 * np.sin(np.pi * progress),
            'intense': 3.0,
            'winter_low': -1.0
        }
        
        return trend_patterns.get(trend, 0.0)
    
    def _apply_constraints(self, param: str, value: float) -> float:
        """Apply realistic constraints to parameter values."""
        constraints = {
            'temperature': (-50.0, 50.0),
            'humidity': (0.0, 100.0),
            'pressure': (920.0, 1080.0),
            'wind_speed': (0.0, 200.0),
            'precipitation': (0.0, 200.0),
            'solar_radiation': (0.0, 1200.0)
        }
        
        if param in constraints:
            min_val, max_val = constraints[param]
            return max(min_val, min(max_val, value))
        
        return value
    
    def check_alert_conditions(self, data_point: Dict, scenario: Dict) -> List[Dict]:
        """Check if current conditions trigger any alerts."""
        alerts = []
        triggers = scenario['alert_triggers']
        
        for trigger_name, threshold in triggers.items():
            alert = self._evaluate_trigger(trigger_name, threshold, data_point, scenario)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _evaluate_trigger(self, trigger_name: str, threshold: float, data_point: Dict, scenario: Dict) -> Optional[Dict]:
        """Evaluate a specific alert trigger."""
        current_time = data_point['time']
        
        # Temperature-based triggers
        if trigger_name == 'critical_temp' and data_point.get('temperature', 0) > threshold:
            return {
                'type': 'CRITICAL TEMPERATURE',
                'severity': 'HIGH',
                'message': f"🚨 CRITICAL: Temperature {data_point['temperature']:.1f}°C exceeds safe limit ({threshold}°C)",
                'time': current_time,
                'value': data_point['temperature'],
                'threshold': threshold,
                'recommendations': ['Activate cooling centers', 'Issue heat warnings', 'Monitor vulnerable populations']
            }
        
        elif trigger_name == 'extreme_temp' and data_point.get('temperature', 0) > threshold:
            return {
                'type': 'EXTREME TEMPERATURE',
                'severity': 'EXTREME',
                'message': f"🔥 EXTREME: Temperature {data_point['temperature']:.1f}°C - LIFE THREATENING CONDITIONS",
                'time': current_time,
                'value': data_point['temperature'],
                'threshold': threshold,
                'recommendations': ['Declare heat emergency', 'Mass cooling center activation', 'Emergency medical standby']
            }
        
        elif trigger_name == 'extreme_cold' and data_point.get('temperature', 0) < threshold:
            return {
                'type': 'EXTREME COLD',
                'severity': 'HIGH',
                'message': f"🥶 EXTREME COLD: Temperature {data_point['temperature']:.1f}°C - Dangerous conditions",
                'time': current_time,
                'value': data_point['temperature'],
                'threshold': threshold,
                'recommendations': ['Open warming shelters', 'Check heating systems', 'Monitor water pipes']
            }
        
        elif trigger_name == 'life_threatening_cold' and data_point.get('temperature', 0) < threshold:
            return {
                'type': 'LIFE THREATENING COLD',
                'severity': 'EXTREME',
                'message': f"🚨 LIFE THREATENING: Temperature {data_point['temperature']:.1f}°C - Immediate action required",
                'time': current_time,
                'value': data_point['temperature'],
                'threshold': threshold,
                'recommendations': ['Declare cold emergency', 'Activate all shelters', 'Emergency outreach teams']
            }
        
        # Pressure-based triggers
        elif trigger_name in ['low_pressure', 'critical_pressure'] and data_point.get('pressure', 1000) < threshold:
            severity = 'EXTREME' if 'critical' in trigger_name else 'HIGH'
            return {
                'type': 'LOW PRESSURE SYSTEM',
                'severity': severity,
                'message': f"🌪️ {'CRITICAL' if severity == 'EXTREME' else 'WARNING'}: Pressure {data_point['pressure']:.1f} hPa - Storm approaching",
                'time': current_time,
                'value': data_point['pressure'],
                'threshold': threshold,
                'recommendations': ['Secure loose objects', 'Avoid outdoor activities', 'Monitor weather updates']
            }
        
        # Wind-based triggers
        elif trigger_name in ['high_wind', 'extreme_wind'] and data_point.get('wind_speed', 0) > threshold:
            severity = 'EXTREME' if 'extreme' in trigger_name else 'HIGH'
            wind_type = 'Föhn' if 'foehn' in trigger_name else 'Storm'
            return {
                'type': f'{wind_type.upper()} WIND WARNING',
                'severity': severity,
                'message': f"💨 {severity}: {wind_type} winds {data_point['wind_speed']:.1f} m/s - Dangerous conditions",
                'time': current_time,
                'value': data_point['wind_speed'],
                'threshold': threshold,
                'recommendations': ['Avoid exposed areas', 'Secure structures', 'Cancel outdoor events']
            }
        
        # Precipitation-based triggers
        elif trigger_name in ['heavy_rain', 'extreme_rain', 'flash_flood_rain'] and data_point.get('precipitation', 0) > threshold:
            severity = 'EXTREME' if 'extreme' in trigger_name or 'flash_flood' in trigger_name else 'HIGH'
            return {
                'type': 'HEAVY PRECIPITATION',
                'severity': severity,
                'message': f"🌊 {'FLASH FLOOD RISK' if 'flash_flood' in trigger_name else 'HEAVY RAIN'}: {data_point['precipitation']:.1f} mm/h",
                'time': current_time,
                'value': data_point['precipitation'],
                'threshold': threshold,
                'recommendations': ['Avoid low-lying areas', 'Monitor water levels', 'Prepare for flooding']
            }
        
        # Humidity-based triggers
        elif trigger_name in ['low_humidity', 'fire_danger_humidity', 'extreme_fire_risk'] and data_point.get('humidity', 50) < threshold:
            return {
                'type': 'EXTREME FIRE RISK',
                'severity': 'EXTREME' if 'extreme' in trigger_name else 'HIGH',
                'message': f"🔥 FIRE DANGER: Humidity {data_point['humidity']:.1f}% - Extreme fire risk conditions",
                'time': current_time,
                'value': data_point['humidity'],
                'threshold': threshold,
                'recommendations': ['No outdoor fires', 'Fire ban in effect', 'Monitor for fire starts']
            }
        
        return None
    
    def run_scenario_simulation(self, scenario_name: str, real_time: bool = False, speed_multiplier: float = 1.0):
        """Run a complete emergency scenario simulation."""
        if scenario_name not in self.scenarios:
            print(f"❌ Unknown scenario: {scenario_name}")
            return
        
        scenario = self.scenarios[scenario_name]
        self.current_scenario = scenario_name
        self.is_running = True
        self.simulation_data = []
        self.alert_history = []
        
        print(f"\n🚨 EMERGENCY SIMULATION STARTING")
        print("=" * 60)
        print(f"📊 Scenario: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        print(f"⏱️  Duration: {scenario['duration_hours']} hours")
        print(f"🎛️  Real-time mode: {'ON' if real_time else 'OFF'}")
        if real_time:
            print(f"⚡ Speed multiplier: {speed_multiplier}x")
        print("=" * 60)
        
        total_alerts = 0
        critical_alerts = 0
        
        for hour in range(scenario['duration_hours']):
            if not self.is_running:
                break
            
            # Generate mock data point
            data_point = self.generate_mock_data_point(scenario, hour)
            self.simulation_data.append(data_point)
            
            # Check for alerts
            alerts = self.check_alert_conditions(data_point, scenario)
            
            if alerts:
                total_alerts += len(alerts)
                for alert in alerts:
                    if alert['severity'] in ['EXTREME', 'CRITICAL']:
                        critical_alerts += 1
                    
                    self.alert_history.append(alert)
                    
                    # Display alert in real-time
                    severity_emoji = "🚨" if alert['severity'] == 'EXTREME' else "⚠️"
                    print(f"\n{severity_emoji} ALERT [{alert['severity']}] - Hour {hour + 1}")
                    print(f"   {alert['message']}")
                    print(f"   Time: {alert['time'].strftime('%H:%M')}")
                    for rec in alert['recommendations'][:2]:  # Show first 2 recommendations
                        print(f"   💡 {rec}")
            
            # Display current conditions
            if hour % 6 == 0 or alerts:  # Show every 6 hours or when there are alerts
                print(f"\n📊 Hour {hour + 1:2d} | {data_point['time'].strftime('%H:%M')}")
                print(f"   🌡️ Temp: {data_point.get('temperature', 0):6.1f}°C")
                print(f"   🌀 Press: {data_point.get('pressure', 0):6.1f} hPa")
                print(f"   💨 Wind: {data_point.get('wind_speed', 0):6.1f} m/s")
                if 'precipitation' in data_point:
                    print(f"   🌧️ Rain: {data_point['precipitation']:6.1f} mm/h")
                print(f"   💧 Humid: {data_point.get('humidity', 0):6.1f}%")
            
            # Real-time delay
            if real_time:
                time.sleep(1.0 / speed_multiplier)
        
        # Simulation complete
        print(f"\n🏁 SIMULATION COMPLETE")
        print("=" * 60)
        print(f"📊 Total alerts generated: {total_alerts}")
        print(f"🚨 Critical/Extreme alerts: {critical_alerts}")
        print(f"⏱️  Scenario duration: {scenario['duration_hours']} hours")
        print("=" * 60)
        
        self.is_running = False
        
        # Save simulation results
        self.save_simulation_results()
        
        return {
            'scenario': scenario_name,
            'duration_hours': scenario['duration_hours'],
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'data_points': len(self.simulation_data),
            'simulation_data': self.simulation_data,
            'alerts': self.alert_history
        }
    
    def save_simulation_results(self):
        """Save simulation results to files."""
        if not self.simulation_data:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        scenario_name = self.current_scenario.replace(' ', '_').lower()
        
        # Save simulation data as CSV
        df = pd.DataFrame(self.simulation_data)
        csv_filename = f"emergency_simulation_{scenario_name}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        # Save alerts as JSON
        alerts_filename = f"emergency_alerts_{scenario_name}_{timestamp}.json"
        with open(alerts_filename, 'w', encoding='utf-8') as f:
            json.dump(self.alert_history, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"💾 Simulation data saved: {csv_filename}")
        print(f"💾 Alert history saved: {alerts_filename}")
    
    def list_scenarios(self):
        """List all available emergency scenarios."""
        print("\n🎭 AVAILABLE EMERGENCY SCENARIOS")
        print("=" * 50)
        
        for key, scenario in self.scenarios.items():
            print(f"🎯 {key}:")
            print(f"   Name: {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            print(f"   Duration: {scenario['duration_hours']} hours")
            print()
    
    def stop_simulation(self):
        """Stop the current simulation."""
        self.is_running = False
        print("🛑 Simulation stopped by user")


def interactive_demo():
    """Interactive demonstration of the emergency simulator."""
    simulator = SwissWeatherEmergencySimulator()
    
    print("🇨🇭 SWISS WEATHER EMERGENCY SIMULATOR")
    print("=" * 60)
    print("🎪 Interactive Emergency Scenario Demonstration")
    print("Perfect for hackathon presentations!")
    print()
    
    while True:
        print("\n🎛️ SIMULATION MENU:")
        print("1. 🔥 Heat Wave Emergency (72 hours)")
        print("2. 🌪️ Severe Storm System (18 hours)")
        print("3. 🌊 Flash Flood Emergency (12 hours)")
        print("4. 🥶 Extreme Cold Emergency (96 hours)")
        print("5. 💨 Föhn Wind Emergency (24 hours)")
        print("6. 📋 List all scenarios")
        print("0. 🚪 Exit")
        
        choice = input("\n🎯 Select scenario (0-6): ").strip()
        
        if choice == '0':
            print("👋 Thank you for using Swiss Weather Emergency Simulator!")
            break
        elif choice == '6':
            simulator.list_scenarios()
            continue
        
        scenario_map = {
            '1': 'heat_wave',
            '2': 'severe_storm', 
            '3': 'flash_flood'
        }
        
        if choice in scenario_map:
            scenario_name = scenario_map[choice]
            
            # Ask for real-time mode
            real_time_input = input("\n⚡ Real-time simulation? (y/N): ").strip().lower()
            real_time = real_time_input in ['y', 'yes']
            
            speed_multiplier = 1.0
            if real_time:
                speed_input = input("🚀 Speed multiplier (1-10, default 5): ").strip()
                try:
                    speed_multiplier = float(speed_input) if speed_input else 5.0
                    speed_multiplier = max(1.0, min(10.0, speed_multiplier))
                except ValueError:
                    speed_multiplier = 5.0
            
            print(f"\n🚀 Starting {simulator.scenarios[scenario_name]['name']}...")
            print("Press Ctrl+C to stop simulation early")
            
            try:
                results = simulator.run_scenario_simulation(scenario_name, real_time, speed_multiplier)
                
                print(f"\n📈 SIMULATION SUMMARY:")
                print(f"   📊 Data points generated: {results['data_points']}")
                print(f"   🚨 Total alerts: {results['total_alerts']}")
                print(f"   ⚠️  Critical alerts: {results['critical_alerts']}")
                
            except KeyboardInterrupt:
                simulator.stop_simulation()
                print("\n⏹️ Simulation interrupted by user")
        
        else:
            print("❌ Invalid choice. Please select 0-6.")


def quick_demo():
    """Quick demonstration of all scenarios."""
    simulator = SwissWeatherEmergencySimulator()
    
    print("🇨🇭 SWISS WEATHER EMERGENCY SIMULATOR")
    print("=" * 60)
    print("🎪 QUICK DEMO - All Emergency Scenarios")
    print()
    
    scenarios_to_demo = ['heat_wave', 'severe_storm', 'flash_flood']
    
    for scenario_name in scenarios_to_demo:
        print(f"\n🎭 Demonstrating: {simulator.scenarios[scenario_name]['name']}")
        
        # Run a short version (12 hours max)
        original_duration = simulator.scenarios[scenario_name]['duration_hours']
        simulator.scenarios[scenario_name]['duration_hours'] = min(12, original_duration)
        
        results = simulator.run_scenario_simulation(scenario_name, real_time=False)
        
        # Restore original duration
        simulator.scenarios[scenario_name]['duration_hours'] = original_duration
        
        print(f"✅ Demo complete: {results['total_alerts']} alerts generated")
        time.sleep(2)
    
    print("\n🏆 ALL SCENARIOS DEMONSTRATED!")
    print("🎯 Emergency simulation system ready for deployment!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_demo()
    else:
        interactive_demo()