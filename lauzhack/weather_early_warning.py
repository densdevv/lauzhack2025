#!/usr/bin/env python3
"""
Swiss Weather Early Warning System
==================================

Predictive early warning system for extreme weather events in Switzerland.
Uses time series forecasting and trend analysis to predict anomalies 
before they become critical.

Features:
- Short-term weather forecasting (6-24 hours ahead)
- Trend-based anomaly prediction
- Risk scoring for different weather hazards
- Real-time alert generation
- Integration with Swiss weather monitoring network

Author: EPFL Hackathon Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    from scipy.signal import savgol_filter
    from scipy.stats import pearsonr
except ImportError as e:
    print(f"⚠️  Missing packages: {e}")

warnings.filterwarnings('ignore')

@dataclass
class WeatherForecast:
    """Weather forecast data structure"""
    timestamp: datetime
    parameter: str
    predicted_value: float
    confidence: float
    risk_level: str
    description: str

@dataclass
class EarlyWarning:
    """Early warning alert structure"""
    alert_id: str
    timestamp: datetime
    event_type: str
    severity: str
    risk_score: float
    affected_parameters: List[str]
    predicted_onset: datetime
    description: str
    recommendations: List[str]


class SwissWeatherEarlyWarning:
    """
    Early warning system for extreme weather prediction in Switzerland.
    
    Capabilities:
    - Predict weather anomalies 6-24 hours in advance
    - Identify developing extreme weather patterns
    - Generate risk scores for different hazard types
    - Provide actionable recommendations
    """
    
    def __init__(self):
        """Initialize the early warning system."""
        self.data = {}
        self.models = {}
        self.forecasts = {}
        self.warnings = []
        
        # Weather hazard definitions for Switzerland
        self.hazard_definitions = {
            'heatwave': {
                'parameters': ['tre200h0'],
                'conditions': {'tre200h0': {'threshold': 30, 'duration': 72}},
                'description': 'Extended period of high temperatures'
            },
            'cold_snap': {
                'parameters': ['tre200h0'],
                'conditions': {'tre200h0': {'threshold': -15, 'duration': 48}},
                'description': 'Extended period of extreme cold'
            },
            'pressure_drop': {
                'parameters': ['prestah0'],
                'conditions': {'prestah0': {'drop_rate': -5, 'duration': 12}},
                'description': 'Rapid atmospheric pressure drop indicating storm approach'
            },
            'wind_storm': {
                'parameters': ['fkl010h0'],
                'conditions': {'fkl010h0': {'threshold': 20, 'duration': 6}},
                'description': 'High wind speeds indicating storm conditions'
            },
            'heavy_precipitation': {
                'parameters': ['rre150h0'],
                'conditions': {'rre150h0': {'cumulative': 50, 'duration': 24}},
                'description': 'Heavy rainfall that could cause flooding'
            }
        }
        
        # Risk scoring weights
        self.risk_weights = {
            'trend_severity': 0.3,
            'rate_of_change': 0.25,
            'historical_rarity': 0.2,
            'seasonal_context': 0.15,
            'multi_parameter': 0.1
        }
    
    def load_weather_data(self, station_code: str) -> bool:
        """Load and prepare weather data for forecasting."""
        try:
            url = f"https://raw.githubusercontent.com/Swiss-ai-Weeks/Building-Resilience-to-Extreme-Weather-in-Switzerland/main/data/ogd-smn_{station_code}_h_recent.csv"
            df = pd.read_csv(url, sep=';')
            
            # Convert timestamp and sort
            df['timestamp'] = pd.to_datetime(df['reference_timestamp'], format='%d.%m.%Y %H:%M')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add time-based features for modeling
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['month'] = df['timestamp'].dt.month
            
            self.data[station_code] = df
            print(f"✅ Loaded {len(df)} records for station {station_code.upper()}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def train_forecasting_models(self, station_code: str, lookback_hours: int = 72) -> Dict:
        """
        Train simple forecasting models for weather parameters.
        
        Args:
            station_code (str): Station code
            lookback_hours (int): Hours of historical data to use for prediction
            
        Returns:
            Dict: Model performance metrics
        """
        if station_code not in self.data:
            return {}
        
        df = self.data[station_code].copy()
        model_performance = {}
        
        print(f"🤖 Training forecasting models for station {station_code.upper()}...")
        
        # Key parameters to forecast
        forecast_params = ['tre200h0', 'prestah0', 'rre150h0', 'fkl010h0', 'ure200h0']
        
        for param in forecast_params:
            if param not in df.columns:
                continue
                
            # Prepare data
            param_data = df[['timestamp', param, 'hour', 'day_of_year']].dropna()
            if len(param_data) < lookback_hours * 2:
                continue
            
            # Create features and targets
            X_features = []
            y_targets = []
            
            for i in range(lookback_hours, len(param_data) - 6):  # Predict 6 hours ahead
                # Features: last N hours + time features
                lookback_values = param_data[param].iloc[i-lookback_hours:i].values
                time_features = [param_data['hour'].iloc[i], param_data['day_of_year'].iloc[i]]
                
                # Add trend features
                recent_trend = np.polyfit(range(24), param_data[param].iloc[i-24:i].values, 1)[0]
                
                features = np.concatenate([
                    [np.mean(lookback_values[-24:])],  # Recent mean
                    [np.std(lookback_values[-24:])],   # Recent std
                    [recent_trend],                    # Recent trend
                    time_features                      # Time features
                ])
                
                X_features.append(features)
                y_targets.append(param_data[param].iloc[i+6])  # Target 6 hours ahead
            
            if len(X_features) < 100:
                continue
                
            X = np.array(X_features)
            y = np.array(y_targets)
            
            # Split into train/test
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            predictions = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, predictions)
            correlation = pearsonr(y_test, predictions)[0]
            
            # Store model
            self.models[f"{station_code}_{param}"] = {
                'model': model,
                'scaler': scaler,
                'lookback_hours': lookback_hours,
                'mae': mae,
                'correlation': correlation
            }
            
            model_performance[param] = {
                'mae': mae,
                'correlation': correlation,
                'rmse': np.sqrt(np.mean((y_test - predictions) ** 2))
            }
            
            print(f"   {param}: MAE={mae:.2f}, Correlation={correlation:.3f}")
        
        return model_performance
    
    def generate_forecasts(self, station_code: str, hours_ahead: int = 24) -> List[WeatherForecast]:
        """
        Generate weather forecasts for the next N hours.
        
        Args:
            station_code (str): Station code
            hours_ahead (int): Hours to forecast ahead
            
        Returns:
            List[WeatherForecast]: List of weather forecasts
        """
        if station_code not in self.data:
            return []
        
        df = self.data[station_code].copy()
        forecasts = []
        
        print(f"🔮 Generating {hours_ahead}-hour forecasts for station {station_code.upper()}...")
        
        # Get most recent data point
        latest_data = df.iloc[-1]
        current_time = latest_data['timestamp']
        
        for param in ['tre200h0', 'prestah0', 'rre150h0', 'fkl010h0', 'ure200h0']:
            model_key = f"{station_code}_{param}"
            if model_key not in self.models:
                continue
            
            model_info = self.models[model_key]
            model = model_info['model']
            scaler = model_info['scaler']
            lookback_hours = model_info['lookback_hours']
            
            # Generate forecast for next 6-hour interval
            forecast_time = current_time + timedelta(hours=6)
            
            # Prepare features
            param_data = df[['timestamp', param, 'hour', 'day_of_year']].dropna()
            recent_values = param_data[param].iloc[-lookback_hours:].values
            
            # Time features for forecast time
            forecast_hour = forecast_time.hour
            forecast_day_of_year = forecast_time.timetuple().tm_yday
            
            # Trend calculation
            recent_trend = np.polyfit(range(24), param_data[param].iloc[-24:].values, 1)[0]
            
            features = np.array([[
                np.mean(recent_values[-24:]),
                np.std(recent_values[-24:]),
                recent_trend,
                forecast_hour,
                forecast_day_of_year
            ]])
            
            # Make prediction
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence based on model performance
            confidence = max(0.1, 1.0 - (model_info['mae'] / np.std(param_data[param])))
            
            # Assess risk level
            risk_level = self._assess_parameter_risk(param, prediction, param_data[param])
            
            forecast = WeatherForecast(
                timestamp=forecast_time,
                parameter=param,
                predicted_value=prediction,
                confidence=confidence,
                risk_level=risk_level,
                description=self._get_forecast_description(param, prediction, risk_level)
            )
            
            forecasts.append(forecast)
        
        self.forecasts[station_code] = forecasts
        return forecasts
    
    def _assess_parameter_risk(self, param: str, predicted_value: float, historical_data: pd.Series) -> str:
        """Assess risk level for a predicted parameter value."""
        # Calculate percentiles from historical data
        p95 = historical_data.quantile(0.95)
        p90 = historical_data.quantile(0.90)
        p10 = historical_data.quantile(0.10)
        p5 = historical_data.quantile(0.05)
        
        if predicted_value >= p95 or predicted_value <= p5:
            return "HIGH"
        elif predicted_value >= p90 or predicted_value <= p10:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_forecast_description(self, param: str, value: float, risk_level: str) -> str:
        """Generate human-readable forecast description."""
        param_names = {
            'tre200h0': f'Temperature: {value:.1f}°C',
            'prestah0': f'Air Pressure: {value:.1f} hPa',
            'rre150h0': f'Precipitation: {value:.1f} mm',
            'fkl010h0': f'Wind Speed: {value:.1f} m/s',
            'ure200h0': f'Humidity: {value:.1f}%'
        }
        
        base_desc = param_names.get(param, f'{param}: {value:.1f}')
        
        if risk_level == "HIGH":
            return f"⚠️ {base_desc} (EXTREME)"
        elif risk_level == "MEDIUM":
            return f"🔶 {base_desc} (ELEVATED)"
        else:
            return f"✅ {base_desc} (NORMAL)"
    
    def detect_developing_hazards(self, station_code: str) -> List[EarlyWarning]:
        """
        Detect developing weather hazards based on current trends and forecasts.
        
        Args:
            station_code (str): Station code
            
        Returns:
            List[EarlyWarning]: List of early warning alerts
        """
        if station_code not in self.data or station_code not in self.forecasts:
            return []
        
        df = self.data[station_code].copy()
        forecasts = self.forecasts[station_code]
        warnings = []
        
        print(f"🔍 Detecting developing hazards for station {station_code.upper()}...")
        
        current_time = datetime.now()
        
        # Check each hazard type
        for hazard_type, hazard_def in self.hazard_definitions.items():
            risk_score = self._calculate_hazard_risk(df, forecasts, hazard_def)
            
            if risk_score > 0.6:  # High risk threshold
                severity = "CRITICAL" if risk_score > 0.8 else "WARNING"
                
                warning = EarlyWarning(
                    alert_id=f"{station_code}_{hazard_type}_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    event_type=hazard_type,
                    severity=severity,
                    risk_score=risk_score,
                    affected_parameters=hazard_def['parameters'],
                    predicted_onset=current_time + timedelta(hours=6),
                    description=hazard_def['description'],
                    recommendations=self._get_hazard_recommendations(hazard_type, severity)
                )
                
                warnings.append(warning)
        
        self.warnings = warnings
        print(f"   Generated {len(warnings)} early warning alerts")
        return warnings
    
    def _calculate_hazard_risk(self, df: pd.DataFrame, forecasts: List[WeatherForecast], 
                              hazard_def: Dict) -> float:
        """Calculate risk score for a specific hazard type."""
        risk_components = {
            'trend_severity': 0.0,
            'rate_of_change': 0.0,
            'historical_rarity': 0.0,
            'seasonal_context': 0.0,
            'multi_parameter': 0.0
        }
        
        # Analyze each parameter involved in the hazard
        for param in hazard_def['parameters']:
            if param not in df.columns:
                continue
                
            recent_data = df[param].dropna().iloc[-72:]  # Last 72 hours
            if len(recent_data) < 24:
                continue
            
            # 1. Trend severity
            if 'threshold' in hazard_def['conditions'].get(param, {}):
                threshold = hazard_def['conditions'][param]['threshold']
                current_value = recent_data.iloc[-1]
                risk_components['trend_severity'] = max(risk_components['trend_severity'],
                                                      abs(current_value - threshold) / threshold)
            
            # 2. Rate of change
            if len(recent_data) >= 6:
                recent_change = recent_data.iloc[-1] - recent_data.iloc[-6]
                historical_std = recent_data.std()
                if historical_std > 0:
                    change_score = abs(recent_change) / (historical_std * 2)
                    risk_components['rate_of_change'] = max(risk_components['rate_of_change'], 
                                                          min(1.0, change_score))
            
            # 3. Historical rarity
            percentile = (recent_data.iloc[-1] < df[param].dropna()).mean()
            if percentile > 0.95 or percentile < 0.05:
                risk_components['historical_rarity'] = 0.8
            elif percentile > 0.90 or percentile < 0.10:
                risk_components['historical_rarity'] = 0.5
        
        # 4. Seasonal context (simplified)
        current_month = df['timestamp'].iloc[-1].month
        if current_month in [12, 1, 2]:  # Winter
            risk_components['seasonal_context'] = 0.3
        elif current_month in [6, 7, 8]:  # Summer
            risk_components['seasonal_context'] = 0.4
        
        # 5. Multi-parameter correlation
        if len(hazard_def['parameters']) > 1:
            risk_components['multi_parameter'] = 0.2
        
        # Calculate weighted risk score
        total_risk = sum(risk_components[comp] * self.risk_weights[comp] 
                        for comp in risk_components)
        
        return min(1.0, total_risk)
    
    def _get_hazard_recommendations(self, hazard_type: str, severity: str) -> List[str]:
        """Get actionable recommendations for a specific hazard."""
        recommendations = {
            'heatwave': [
                "Stay hydrated and avoid prolonged sun exposure",
                "Check on elderly and vulnerable populations",
                "Avoid strenuous outdoor activities during peak hours",
                "Ensure air conditioning systems are functioning"
            ],
            'cold_snap': [
                "Protect exposed skin and dress in layers",
                "Check heating systems and ensure adequate fuel supply",
                "Protect pipes from freezing",
                "Monitor for signs of hypothermia"
            ],
            'pressure_drop': [
                "Secure loose outdoor objects",
                "Monitor weather updates closely",
                "Avoid unnecessary travel",
                "Prepare for possible power outages"
            ],
            'wind_storm': [
                "Stay indoors and away from windows",
                "Avoid travel, especially in high-profile vehicles",
                "Report downed power lines immediately",
                "Secure or bring indoors loose outdoor items"
            ],
            'heavy_precipitation': [
                "Avoid driving through flooded roads",
                "Stay away from storm drains and waterways",
                "Monitor local flood warnings",
                "Prepare emergency supplies"
            ]
        }
        
        base_recs = recommendations.get(hazard_type, ["Monitor conditions closely"])
        
        if severity == "CRITICAL":
            base_recs.insert(0, "IMMEDIATE ACTION REQUIRED")
            base_recs.append("Consider evacuation if advised by authorities")
        
        return base_recs
    
    def create_early_warning_dashboard(self, station_code: str):
        """Create comprehensive early warning dashboard."""
        if station_code not in self.data:
            return
        
        df = self.data[station_code].copy()
        forecasts = self.forecasts.get(station_code, [])
        warnings = self.warnings
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'🇨🇭 Swiss Weather Early Warning System - Station {station_code.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # Plot recent trends and forecasts
        params = ['tre200h0', 'prestah0', 'rre150h0', 'fkl010h0', 'ure200h0']
        param_names = ['Temperature (°C)', 'Pressure (hPa)', 'Precipitation (mm)', 
                      'Wind Speed (m/s)', 'Humidity (%)']
        
        for i, (param, name) in enumerate(zip(params, param_names)):
            row, col = i // 3, i % 3
            if row >= 2:
                break
                
            ax = axes[row, col]
            
            if param in df.columns:
                # Plot recent data (last 7 days)
                recent_data = df[['timestamp', param]].dropna().iloc[-168:]  # Last week
                ax.plot(recent_data['timestamp'], recent_data[param], 'b-', 
                       linewidth=1, alpha=0.7, label='Historical')
                
                # Plot forecast
                forecast_data = [f for f in forecasts if f.parameter == param]
                if forecast_data:
                    forecast = forecast_data[0]
                    ax.scatter([forecast.timestamp], [forecast.predicted_value], 
                             color='red', s=100, marker='*', 
                             label=f'Forecast ({forecast.risk_level})')
                
                ax.set_title(name)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.tick_params(axis='x', rotation=45)
        
        # Warning summary in the last subplot
        ax = axes[1, 2]
        if warnings:
            warning_counts = {}
            for w in warnings:
                warning_counts[w.severity] = warning_counts.get(w.severity, 0) + 1
            
            colors = {'CRITICAL': 'red', 'WARNING': 'orange', 'INFO': 'yellow'}
            ax.pie(warning_counts.values(), labels=warning_counts.keys(), autopct='%1.0f',
                   colors=[colors.get(k, 'gray') for k in warning_counts.keys()])
            ax.set_title(f'Active Warnings ({len(warnings)} total)')
        else:
            ax.text(0.5, 0.5, 'No Active Warnings', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='green')
            ax.set_title('Warning Status')
        
        plt.tight_layout()
        plt.savefig(f'early_warning_dashboard_{station_code}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Early warning dashboard saved as 'early_warning_dashboard_{station_code}.png'")


def main():
    """Main function to demonstrate the early warning system."""
    print("🇨🇭 Swiss Weather Early Warning System")
    print("=" * 50)
    
    # Initialize system
    warning_system = SwissWeatherEarlyWarning()
    
    # Load data
    station = 'beh'
    if not warning_system.load_weather_data(station):
        return
    
    # Train forecasting models
    print(f"\n🤖 Training forecasting models...")
    performance = warning_system.train_forecasting_models(station)
    
    # Generate forecasts
    print(f"\n🔮 Generating weather forecasts...")
    forecasts = warning_system.generate_forecasts(station, hours_ahead=24)
    
    # Detect developing hazards
    print(f"\n🔍 Detecting developing weather hazards...")
    warnings = warning_system.detect_developing_hazards(station)
    
    # Create dashboard
    print(f"\n📊 Creating early warning dashboard...")
    warning_system.create_early_warning_dashboard(station)
    
    # Print forecast summary
    print(f"\n🔮 WEATHER FORECAST SUMMARY")
    print("=" * 30)
    for forecast in forecasts:
        print(f"{forecast.description} (Confidence: {forecast.confidence:.1%})")
    
    # Print warning summary
    if warnings:
        print(f"\n⚠️  EARLY WARNING ALERTS")
        print("=" * 25)
        for warning in warnings:
            print(f"🚨 {warning.severity}: {warning.event_type.upper()}")
            print(f"   Risk Score: {warning.risk_score:.1%}")
            print(f"   {warning.description}")
            print(f"   Recommendations:")
            for rec in warning.recommendations[:3]:
                print(f"   • {rec}")
            print()
    else:
        print(f"\n✅ No immediate weather hazards detected")
    
    print(f"\n✅ Early warning system operational!")
    print(f"🎯 Ready for real-time extreme weather prediction!")


if __name__ == "__main__":
    main()