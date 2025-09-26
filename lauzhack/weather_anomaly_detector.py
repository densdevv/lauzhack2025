#!/usr/bin/env python3
"""
Swiss Weather Anomaly Detection System
======================================

Advanced anomaly detection for Swiss meteorological data to predict
extreme weather events and build resilience against climate risks.

Features:
- Statistical anomaly detection using Isolation Forest and Z-scores
- Change point detection for abrupt parameter shifts
- Time series trend analysis and seasonal decomposition
- Early warning system for extreme weather conditions
- Interactive visualizations and alerts

Author: EPFL Hackathon Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass

# ML and statistical libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score, silhouette_score
    from sklearn.model_selection import ParameterGrid
    from scipy import stats
    from scipy.signal import find_peaks
    import matplotlib.dates as mdates
except ImportError as e:
    print(f"⚠️  Missing required packages. Please install: {e}")
    print("Run: pip install scikit-learn scipy matplotlib seaborn")

warnings.filterwarnings('ignore')

@dataclass
class AnomalyAlert:
    """Data class for anomaly alerts"""
    timestamp: str
    parameter: str
    value: float
    anomaly_type: str
    severity: str
    description: str
    threshold: float


class SwissWeatherAnomalyDetector:
    """
    Advanced anomaly detection system for Swiss weather data.
    
    Detects:
    - Statistical outliers using Isolation Forest
    - Sudden changes and trend breaks
    - Extreme values beyond historical norms
    - Seasonal anomalies
    """
    
    def __init__(self):
        """Initialize the anomaly detection system."""
        self.data = {}
        self.station_data = {}  # For enhanced validation
        self.anomalies = []
        self.alerts = []
        
        # Critical weather parameters for monitoring (support both Swiss codes and simple names)
        self.critical_params = {
            # Swiss meteorological codes
            'tre200h0': {'name': 'Temperature', 'unit': '°C', 'extreme_threshold': 3.0},
            'prestah0': {'name': 'Air Pressure', 'unit': 'hPa', 'extreme_threshold': 2.5},
            'rre150h0': {'name': 'Precipitation', 'unit': 'mm', 'extreme_threshold': 2.0},
            'fkl010h0': {'name': 'Wind Speed', 'unit': 'm/s', 'extreme_threshold': 2.5},
            'ure200h0': {'name': 'Humidity', 'unit': '%', 'extreme_threshold': 2.0},
            'gre000h0': {'name': 'Solar Radiation', 'unit': 'W/m²', 'extreme_threshold': 2.0},
            # Simple names for emergency simulation data
            'temperature': {'name': 'Temperature', 'unit': '°C', 'extreme_threshold': 3.0},
            'pressure': {'name': 'Air Pressure', 'unit': 'hPa', 'extreme_threshold': 2.5},
            'humidity': {'name': 'Humidity', 'unit': '%', 'extreme_threshold': 2.0},
            'wind_speed': {'name': 'Wind Speed', 'unit': 'm/s', 'extreme_threshold': 2.5},
            'solar_radiation': {'name': 'Solar Radiation', 'unit': 'W/m²', 'extreme_threshold': 2.0}
        }
        
        # Extreme weather thresholds for Switzerland
        self.extreme_thresholds = {
            'tre200h0': {'low': -20, 'high': 35, 'critical_low': -25, 'critical_high': 40},
            'prestah0': {'low': 950, 'high': 1050, 'critical_low': 940, 'critical_high': 1060},
            'rre150h0': {'high': 20, 'critical_high': 40},  # mm/hour
            'fkl010h0': {'high': 15, 'critical_high': 25},  # m/s
            'ure200h0': {'low': 10, 'high': 95, 'critical_low': 5, 'critical_high': 98}
        }
    
    def load_station_data(self, station_code: str) -> bool:
        """
        Load weather data for a specific station.
        
        Args:
            station_code (str): Station code (e.g., 'beh', 'int', 'zer')
            
        Returns:
            bool: Success status
        """
        try:
            url = f"https://raw.githubusercontent.com/Swiss-ai-Weeks/Building-Resilience-to-Extreme-Weather-in-Switzerland/main/data/ogd-smn_{station_code}_h_recent.csv"
            df = pd.read_csv(url, sep=';')
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['reference_timestamp'], format='%d.%m.%Y %H:%M')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Store data
            self.data[station_code] = df
            
            print(f"✅ Loaded {len(df)} records for station {station_code.upper()}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load data for station {station_code}: {e}")
            return False
    
    def detect_statistical_anomalies(self, station_code: str, contamination: float = 0.1) -> Dict:
        """
        Detect statistical anomalies using Isolation Forest and Z-scores.
        
        Args:
            station_code (str): Station code
            contamination (float): Expected proportion of anomalies
            
        Returns:
            Dict: Anomaly detection results
        """
        if station_code not in self.data:
            print(f"❌ No data loaded for station {station_code}")
            return {}
        
        df = self.data[station_code].copy()
        results = {}
        
        print(f"🔍 Detecting statistical anomalies for station {station_code.upper()}...")
        
        for param in self.critical_params.keys():
            if param not in df.columns:
                continue
                
            # Remove missing values
            param_data = df[['timestamp', param]].dropna()
            if len(param_data) < 100:  # Need sufficient data
                continue
                
            values = param_data[param].values.reshape(-1, 1)
            
            # Standardize data
            scaler = StandardScaler()
            values_scaled = scaler.fit_transform(values)
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(values_scaled)
            
            # Z-score analysis
            z_scores = np.abs(stats.zscore(values.flatten()))
            z_threshold = self.critical_params[param]['extreme_threshold']
            z_anomalies = z_scores > z_threshold
            
            # Combine both methods
            combined_anomalies = (anomaly_labels == -1) | z_anomalies
            
            # Store results
            anomaly_indices = param_data.index[combined_anomalies]
            anomaly_timestamps = param_data.loc[anomaly_indices, 'timestamp']
            anomaly_values = param_data.loc[anomaly_indices, param]
            
            results[param] = {
                'name': self.critical_params[param]['name'],
                'unit': self.critical_params[param]['unit'],
                'total_anomalies': int(combined_anomalies.sum()),
                'anomaly_rate': float(combined_anomalies.sum() / len(param_data) * 100),
                'timestamps': anomaly_timestamps.tolist(),
                'values': anomaly_values.tolist(),
                'z_scores': z_scores[combined_anomalies].tolist(),
                'indices': anomaly_indices.tolist()
            }
            
            print(f"   {param}: {results[param]['total_anomalies']} anomalies ({results[param]['anomaly_rate']:.1f}%)")
        
        return results
    
    def detect_change_points(self, station_code: str, window_size: int = 24) -> Dict:
        """
        Detect abrupt changes and trend breaks in weather parameters.
        
        Args:
            station_code (str): Station code
            window_size (int): Rolling window size for change detection
            
        Returns:
            Dict: Change point detection results
        """
        if station_code not in self.data:
            return {}
        
        df = self.data[station_code].copy()
        results = {}
        
        print(f"📈 Detecting change points for station {station_code.upper()}...")
        
        for param in self.critical_params.keys():
            if param not in df.columns:
                continue
                
            param_data = df[['timestamp', param]].dropna()
            if len(param_data) < window_size * 3:
                continue
            
            values = param_data[param].values
            
            # Calculate rolling statistics
            rolling_mean = pd.Series(values).rolling(window=window_size).mean()
            rolling_std = pd.Series(values).rolling(window=window_size).std()
            
            # Detect significant changes in rolling mean
            mean_diff = np.abs(np.diff(rolling_mean.dropna()))
            mean_threshold = np.percentile(mean_diff, 95)  # Top 5% changes
            
            # Find change points
            change_points = find_peaks(mean_diff, height=mean_threshold)[0]
            
            # Adjust indices for rolling window offset
            change_indices = change_points + window_size
            change_indices = change_indices[change_indices < len(param_data)]
            
            results[param] = {
                'name': self.critical_params[param]['name'],
                'unit': self.critical_params[param]['unit'],
                'change_points': len(change_indices),
                'timestamps': param_data.iloc[change_indices]['timestamp'].tolist(),
                'values': param_data.iloc[change_indices][param].tolist(),
                'indices': change_indices.tolist()
            }
            
            print(f"   {param}: {results[param]['change_points']} change points")
        
        return results
    
    def detect_anomalies_with_validation(self, station_code: str) -> Dict:
        """
        Enhanced anomaly detection with proper train/test splitting and multiple evaluation methods.
        Automatically selects the best performing approach based on comprehensive metrics.
        
        Args:
            station_code (str): Station code
            
        Returns:
            Dict: Comprehensive anomaly detection results with performance metrics
        """
        if station_code not in self.station_data:
            print(f"❌ No data available for station {station_code}")
            return {}
        
        data = self.station_data[station_code].copy()
        if data.empty:
            return {}
        
        print(f"\n🧪 Enhanced Anomaly Detection with Validation - Station {station_code.upper()}")
        print("=" * 70)
        
        all_results = {}
        best_models = {}
        
        for param in self.critical_params.keys():
            if param not in data.columns:
                continue
                
            print(f"\n📊 Analyzing {self.critical_params[param]['name']}...")
            
            # Prepare data
            param_data = data[['timestamp', param]].dropna()
            if len(param_data) < 20:  # Need sufficient data (minimal for demo)
                print(f"   ⚠️  Insufficient data ({len(param_data)} points)")
                continue
            
            # Chronological split (important for time series)
            split_idx = int(0.7 * len(param_data))
            train_data = param_data.iloc[:split_idx]
            test_data = param_data.iloc[split_idx:]
            
            # Prepare features
            X_train = train_data[param].values.reshape(-1, 1)
            X_test = test_data[param].values.reshape(-1, 1)
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test multiple approaches
            approaches = self._evaluate_anomaly_approaches(X_train_scaled, X_test_scaled, param)
            
            # Select best approach
            best_approach = max(approaches, key=lambda x: x['composite_score'])
            best_models[param] = best_approach
            
            # Apply best model to full dataset for final results
            X_full_scaled = scaler.fit_transform(param_data[param].values.reshape(-1, 1))
            final_anomalies = best_approach['model'].predict(X_full_scaled)
            
            # Combine with statistical analysis
            z_scores = np.abs(stats.zscore(param_data[param].values))
            z_threshold = self.critical_params[param]['extreme_threshold']
            statistical_anomalies = z_scores > z_threshold
            
            # Final anomaly detection (ML + Statistical)
            ml_anomalies = (final_anomalies == -1)
            combined_anomalies = ml_anomalies | statistical_anomalies
            
            # Store comprehensive results
            anomaly_indices = param_data.index[combined_anomalies]
            all_results[param] = {
                'name': self.critical_params[param]['name'],
                'unit': self.critical_params[param]['unit'],
                'best_model': best_approach['name'],
                'model_performance': {
                    'precision': best_approach['precision'],
                    'recall': best_approach['recall'],
                    'f1_score': best_approach['f1_score'],
                    'silhouette_score': best_approach['silhouette_score'],
                    'composite_score': best_approach['composite_score']
                },
                'total_anomalies': int(combined_anomalies.sum()),
                'ml_anomalies': int(ml_anomalies.sum()),
                'statistical_anomalies': int(statistical_anomalies.sum()),
                'anomaly_rate': float(combined_anomalies.sum() / len(param_data) * 100),
                'timestamps': param_data.loc[anomaly_indices, 'timestamp'].tolist(),
                'values': param_data.loc[anomaly_indices, param].tolist(),
                'z_scores': z_scores[combined_anomalies].tolist(),
                'confidence_scores': best_approach.get('confidence_scores', []),
                'indices': anomaly_indices.tolist()
            }
            
            print(f"   ✅ Best Model: {best_approach['name']}")
            print(f"   📈 Performance: F1={best_approach['f1_score']:.3f}, Precision={best_approach['precision']:.3f}")
            print(f"   🔍 Detected: {all_results[param]['total_anomalies']} anomalies ({all_results[param]['anomaly_rate']:.1f}%)")
        
        # Overall summary
        if all_results:
            avg_f1 = np.mean([r['model_performance']['f1_score'] for r in all_results.values()])
            avg_precision = np.mean([r['model_performance']['precision'] for r in all_results.values()])
            total_anomalies = sum([r['total_anomalies'] for r in all_results.values()])
            
            print(f"\n🎯 VALIDATION SUMMARY")
            print(f"   Average F1-Score: {avg_f1:.3f}")
            print(f"   Average Precision: {avg_precision:.3f}")
            print(f"   Total Anomalies Detected: {total_anomalies}")
            
            all_results['summary'] = {
                'average_f1_score': avg_f1,
                'average_precision': avg_precision,
                'total_anomalies': total_anomalies,
                'parameters_analyzed': len(all_results) - 1  # Exclude summary itself
            }
        
        return all_results
    
    def _evaluate_anomaly_approaches(self, X_train: np.ndarray, X_test: np.ndarray, param: str) -> List[Dict]:
        """
        Evaluate multiple anomaly detection approaches and return performance metrics.
        
        Args:
            X_train: Training data
            X_test: Test data  
            param: Parameter name for context
            
        Returns:
            List[Dict]: Performance results for each approach
        """
        approaches = []
        
        # 1. Isolation Forest with multiple contamination rates
        contamination_rates = [0.05, 0.1, 0.15, 0.2]
        for contamination in contamination_rates:
            try:
                model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
                model.fit(X_train)
                
                # Predict on test set
                test_predictions = model.predict(X_test)
                train_predictions = model.predict(X_train)
                
                # Calculate metrics (assuming normal data in training set)
                precision, recall, f1 = self._calculate_anomaly_metrics(X_test, test_predictions)
                silhouette = self._calculate_silhouette_score(X_test, test_predictions)
                
                # Composite score (weighted combination)
                composite_score = 0.4 * f1 + 0.3 * precision + 0.2 * recall + 0.1 * silhouette
                
                approaches.append({
                    'name': f'IsolationForest (contamination={contamination})',
                    'model': model,
                    'contamination': contamination,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'silhouette_score': silhouette,
                    'composite_score': composite_score,
                    'anomaly_count': int((test_predictions == -1).sum())
                })
            except Exception as e:
                print(f"     ⚠️  IsolationForest (contamination={contamination}) failed: {e}")
        
        # 2. One-Class SVM with different parameters
        svm_configs = [
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.2},
            {'kernel': 'linear', 'nu': 0.1}
        ]
        
        for config in svm_configs:
            try:
                model = OneClassSVM(**config)
                model.fit(X_train)
                
                test_predictions = model.predict(X_test)
                
                precision, recall, f1 = self._calculate_anomaly_metrics(X_test, test_predictions)
                silhouette = self._calculate_silhouette_score(X_test, test_predictions)
                
                composite_score = 0.4 * f1 + 0.3 * precision + 0.2 * recall + 0.1 * silhouette
                
                approaches.append({
                    'name': f'OneClassSVM ({config})',
                    'model': model,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'silhouette_score': silhouette,
                    'composite_score': composite_score,
                    'anomaly_count': int((test_predictions == -1).sum())
                })
            except Exception as e:
                print(f"     ⚠️  OneClassSVM {config} failed: {e}")
        
        # 3. Statistical Z-score approach
        try:
            z_scores_test = np.abs(stats.zscore(X_test.flatten()))
            z_threshold = self.critical_params[param]['extreme_threshold']
            statistical_predictions = np.where(z_scores_test > z_threshold, -1, 1)
            
            precision, recall, f1 = self._calculate_anomaly_metrics(X_test, statistical_predictions)
            silhouette = self._calculate_silhouette_score(X_test, statistical_predictions)
            
            composite_score = 0.4 * f1 + 0.3 * precision + 0.2 * recall + 0.1 * silhouette
            
            # Create dummy model for consistency
            class StatisticalModel:
                def __init__(self, threshold):
                    self.threshold = threshold
                def predict(self, X):
                    z_scores = np.abs(stats.zscore(X.flatten()))
                    return np.where(z_scores > self.threshold, -1, 1)
            
            approaches.append({
                'name': f'Statistical Z-score (threshold={z_threshold})',
                'model': StatisticalModel(z_threshold),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'silhouette_score': silhouette,
                'composite_score': composite_score,
                'anomaly_count': int((statistical_predictions == -1).sum())
            })
        except Exception as e:
            print(f"     ⚠️  Statistical approach failed: {e}")
        
        return approaches
    
    def _calculate_anomaly_metrics(self, X_test: np.ndarray, predictions: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1-score for anomaly detection.
        For unsupervised learning, we use extreme values as ground truth.
        """
        try:
            # Use extreme values (top/bottom 10%) as ground truth anomalies
            percentile_90 = np.percentile(X_test, 90)
            percentile_10 = np.percentile(X_test, 10)
            true_anomalies = (X_test.flatten() > percentile_90) | (X_test.flatten() < percentile_10)
            predicted_anomalies = (predictions == -1)
            
            if true_anomalies.sum() == 0 or predicted_anomalies.sum() == 0:
                return 0.0, 0.0, 0.0
            
            precision = precision_score(true_anomalies, predicted_anomalies, zero_division=0)
            recall = recall_score(true_anomalies, predicted_anomalies, zero_division=0)
            f1 = f1_score(true_anomalies, predicted_anomalies, zero_division=0)
            
            return precision, recall, f1
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_silhouette_score(self, X_test: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate silhouette score for anomaly detection clustering quality."""
        try:
            if len(np.unique(predictions)) < 2:  # Need at least 2 clusters
                return 0.0
            # Convert -1, 1 to 0, 1 for silhouette score
            labels = np.where(predictions == -1, 0, 1)
            return silhouette_score(X_test, labels)
        except:
            return 0.0
    
    def detect_extreme_conditions(self, station_code: str) -> List[AnomalyAlert]:
        """
        Detect extreme weather conditions that could pose risks.
        
        Args:
            station_code (str): Station code
            
        Returns:
            List[AnomalyAlert]: List of extreme weather alerts
        """
        if station_code not in self.data:
            return []
        
        df = self.data[station_code].copy()
        alerts = []
        
        print(f"⚠️  Checking for extreme conditions at station {station_code.upper()}...")
        
        for param, thresholds in self.extreme_thresholds.items():
            if param not in df.columns:
                continue
                
            param_data = df[['timestamp', param]].dropna()
            
            for _, row in param_data.iterrows():
                timestamp = row['timestamp']
                value = row[param]
                
                # Check for extreme conditions
                severity = None
                alert_type = None
                
                if 'critical_low' in thresholds and value <= thresholds['critical_low']:
                    severity = 'CRITICAL'
                    alert_type = 'extreme_low'
                elif 'critical_high' in thresholds and value >= thresholds['critical_high']:
                    severity = 'CRITICAL'
                    alert_type = 'extreme_high'
                elif 'low' in thresholds and value <= thresholds['low']:
                    severity = 'WARNING'
                    alert_type = 'low'
                elif 'high' in thresholds and value >= thresholds['high']:
                    severity = 'WARNING'
                    alert_type = 'high'
                
                if severity:
                    description = self._get_alert_description(param, alert_type, value)
                    threshold = thresholds.get(f'critical_{alert_type.split("_")[-1]}', 
                                             thresholds.get(alert_type.split('_')[-1]))
                    
                    alert = AnomalyAlert(
                        timestamp=timestamp.strftime('%Y-%m-%d %H:%M'),
                        parameter=param,
                        value=value,
                        anomaly_type=alert_type,
                        severity=severity,
                        description=description,
                        threshold=threshold
                    )
                    alerts.append(alert)
        
        # Sort alerts by timestamp
        alerts.sort(key=lambda x: x.timestamp)
        
        print(f"   Found {len(alerts)} extreme weather alerts")
        return alerts
    
    def _get_alert_description(self, param: str, alert_type: str, value: float) -> str:
        """Generate human-readable alert descriptions."""
        param_info = self.critical_params.get(param, {})
        param_name = param_info.get('name', param)
        unit = param_info.get('unit', '')
        
        descriptions = {
            'extreme_low': f"Extremely low {param_name.lower()}: {value}{unit}",
            'extreme_high': f"Extremely high {param_name.lower()}: {value}{unit}",
            'low': f"Low {param_name.lower()}: {value}{unit}",
            'high': f"High {param_name.lower()}: {value}{unit}"
        }
        
        return descriptions.get(alert_type, f"Anomalous {param_name.lower()}: {value}{unit}")
    
    def create_anomaly_dashboard(self, station_code: str, anomaly_results: Dict, 
                                change_results: Dict, alerts: List[AnomalyAlert]):
        """
        Create comprehensive visualization dashboard for anomalies.
        
        Args:
            station_code (str): Station code
            anomaly_results (Dict): Statistical anomaly results
            change_results (Dict): Change point results
            alerts (List[AnomalyAlert]): Extreme weather alerts
        """
        if station_code not in self.data:
            return
            
        df = self.data[station_code].copy()
        
        # Create a comprehensive dashboard
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        fig.suptitle(f'🇨🇭 Swiss Weather Anomaly Detection Dashboard - Station {station_code.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Temperature anomalies
        self._plot_parameter_anomalies(axes[0, 0], df, 'tre200h0', 'Temperature (°C)', 
                                     anomaly_results, change_results)
        
        # Plot 2: Pressure anomalies
        self._plot_parameter_anomalies(axes[0, 1], df, 'prestah0', 'Air Pressure (hPa)', 
                                     anomaly_results, change_results)
        
        # Plot 3: Precipitation anomalies
        self._plot_parameter_anomalies(axes[1, 0], df, 'rre150h0', 'Precipitation (mm)', 
                                     anomaly_results, change_results)
        
        # Plot 4: Wind speed anomalies
        self._plot_parameter_anomalies(axes[1, 1], df, 'fkl010h0', 'Wind Speed (m/s)', 
                                     anomaly_results, change_results)
        
        # Plot 5: Alert summary
        self._plot_alert_summary(axes[2, 0], alerts)
        
        # Plot 6: Anomaly statistics
        self._plot_anomaly_statistics(axes[2, 1], anomaly_results)
        
        plt.tight_layout()
        plt.savefig(f'weather_anomaly_dashboard_{station_code}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Dashboard saved as 'weather_anomaly_dashboard_{station_code}.png'")
    
    def _plot_parameter_anomalies(self, ax, df, param, title, anomaly_results, change_results):
        """Plot anomalies for a specific parameter."""
        if param not in df.columns or param not in anomaly_results:
            ax.set_title(f"{title} - No Data")
            return
            
        # Plot main time series
        param_data = df[['timestamp', param]].dropna()
        ax.plot(param_data['timestamp'], param_data[param], 'b-', alpha=0.7, linewidth=1)
        
        # Highlight anomalies
        anomaly_info = anomaly_results[param]
        if anomaly_info['timestamps']:
            anomaly_times = pd.to_datetime(anomaly_info['timestamps'])
            anomaly_vals = anomaly_info['values']
            ax.scatter(anomaly_times, anomaly_vals, color='red', s=30, alpha=0.8, 
                      label=f"Anomalies ({len(anomaly_vals)})")
        
        # Mark change points
        if param in change_results and change_results[param]['timestamps']:
            change_times = pd.to_datetime(change_results[param]['timestamps'])
            change_vals = change_results[param]['values']
            ax.scatter(change_times, change_vals, color='orange', s=50, marker='^', 
                      alpha=0.8, label=f"Change Points ({len(change_vals)})")
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel(param)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_alert_summary(self, ax, alerts):
        """Plot alert summary."""
        if not alerts:
            ax.text(0.5, 0.5, 'No Extreme Weather Alerts', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Extreme Weather Alerts')
            return
        
        # Count alerts by severity
        severity_counts = {}
        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Create pie chart
        colors = {'CRITICAL': 'red', 'WARNING': 'orange', 'INFO': 'yellow'}
        ax.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%',
               colors=[colors.get(k, 'gray') for k in severity_counts.keys()])
        ax.set_title(f'Alert Summary ({len(alerts)} total)')
    
    def _plot_anomaly_statistics(self, ax, anomaly_results):
        """Plot anomaly statistics."""
        if not anomaly_results:
            ax.text(0.5, 0.5, 'No Anomaly Data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Anomaly Statistics')
            return
        
        # Extract anomaly rates
        params = []
        rates = []
        for param, info in anomaly_results.items():
            params.append(info['name'])
            rates.append(info['anomaly_rate'])
        
        # Create bar chart
        bars = ax.bar(params, rates, color='lightcoral', alpha=0.7)
        ax.set_title('Anomaly Rates by Parameter')
        ax.set_ylabel('Anomaly Rate (%)')
        ax.set_xlabel('Weather Parameter')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def generate_report(self, station_code: str, anomaly_results: Dict, 
                       change_results: Dict, alerts: List[AnomalyAlert]) -> str:
        """Generate a comprehensive anomaly detection report."""
        report = f"""
🇨🇭 SWISS WEATHER ANOMALY DETECTION REPORT
==========================================
Station: {station_code.upper()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 STATISTICAL ANOMALIES
------------------------
"""
        
        for param, info in anomaly_results.items():
            report += f"""
{info['name']} [{info['unit']}]:
  • Total anomalies: {info['total_anomalies']}
  • Anomaly rate: {info['anomaly_rate']:.1f}%
  • Most recent anomaly: {info['timestamps'][-1] if info['timestamps'] else 'None'}
"""
        
        report += f"""
📈 CHANGE POINT ANALYSIS
------------------------
"""
        
        for param, info in change_results.items():
            report += f"""
{info['name']} [{info['unit']}]:
  • Change points detected: {info['change_points']}
  • Most recent change: {info['timestamps'][-1] if info['timestamps'] else 'None'}
"""
        
        report += f"""
⚠️  EXTREME WEATHER ALERTS
--------------------------
Total alerts: {len(alerts)}
"""
        
        # Group alerts by severity
        critical_alerts = [a for a in alerts if a.severity == 'CRITICAL']
        warning_alerts = [a for a in alerts if a.severity == 'WARNING']
        
        if critical_alerts:
            report += f"\n🚨 CRITICAL ALERTS ({len(critical_alerts)}):\n"
            for alert in critical_alerts[-5:]:  # Show last 5
                report += f"  • {alert.timestamp}: {alert.description}\n"
        
        if warning_alerts:
            report += f"\n⚠️  WARNING ALERTS ({len(warning_alerts)}):\n"
            for alert in warning_alerts[-5:]:  # Show last 5
                report += f"  • {alert.timestamp}: {alert.description}\n"
        
        report += f"""

🎯 RECOMMENDATIONS
------------------
Based on the analysis:
"""
        
        if critical_alerts:
            report += "• IMMEDIATE ACTION REQUIRED: Critical weather conditions detected\n"
        if len(alerts) > 10:
            report += "• Enhanced monitoring recommended due to frequent extreme conditions\n"
        
        total_anomalies = sum(info['total_anomalies'] for info in anomaly_results.values())
        if total_anomalies > 100:
            report += "• High anomaly rate suggests unstable weather patterns\n"
        
        report += "• Continue monitoring for early warning of extreme weather events\n"
        
        return report


def main():
    """Main function to demonstrate the anomaly detection system."""
    print("🇨🇭 Swiss Weather Anomaly Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = SwissWeatherAnomalyDetector()
    
    # Load data for a station (BEH - Bern)
    station = 'beh'
    if not detector.load_station_data(station):
        print("❌ Failed to load data. Exiting.")
        return
    
    print(f"\n🔍 Running comprehensive anomaly analysis...")
    
    # Detect statistical anomalies
    anomaly_results = detector.detect_statistical_anomalies(station)
    
    # Detect change points
    change_results = detector.detect_change_points(station)
    
    # Detect extreme conditions
    alerts = detector.detect_extreme_conditions(station)
    
    # Create dashboard
    print(f"\n📊 Creating anomaly detection dashboard...")
    detector.create_anomaly_dashboard(station, anomaly_results, change_results, alerts)
    
    # Generate report
    report = detector.generate_report(station, anomaly_results, change_results, alerts)
    
    # Save report
    with open(f'weather_anomaly_report_{station}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 Report saved as 'weather_anomaly_report_{station}.txt'")
    print(report)
    
    print(f"\n✅ Anomaly detection complete!")
    print(f"🎯 Ready for extreme weather prediction and early warning systems!")


if __name__ == "__main__":
    main()