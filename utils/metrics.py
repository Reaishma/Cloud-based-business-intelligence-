import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score, learning_curve
import random

class MetricsCalculator:
    """Calculate and provide ML model performance metrics"""
    
    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def get_model_performance(self, model_name):
        """Get high-level performance metrics for a model"""
        
        # Generate realistic performance metrics based on model type
        if "Sales Forecasting" in model_name:
            base_accuracy = np.random.uniform(0.88, 0.95)
        elif "Churn Prediction" in model_name:
            base_accuracy = np.random.uniform(0.85, 0.93)
        elif "Customer Segmentation" in model_name:
            base_accuracy = np.random.uniform(0.82, 0.90)
        else:
            base_accuracy = np.random.uniform(0.80, 0.92)
        
        return {
            'accuracy': base_accuracy,
            'precision': base_accuracy * np.random.uniform(0.95, 1.05),
            'recall': base_accuracy * np.random.uniform(0.90, 1.02),
            'f1_score': base_accuracy * np.random.uniform(0.92, 1.03),
            'accuracy_delta': np.random.uniform(-0.02, 0.03),
            'precision_delta': np.random.uniform(-0.01, 0.02),
            'recall_delta': np.random.uniform(-0.02, 0.01),
            'f1_delta': np.random.uniform(-0.01, 0.02)
        }
    
    def get_detailed_metrics(self, model_name):
        """Get detailed performance metrics including curves and matrices"""
        
        if "Churn Prediction" in model_name or "Customer Segmentation" in model_name:
            return self._get_classification_metrics()
        else:
            return self._get_regression_metrics()
    
    def _get_classification_metrics(self):
        """Generate classification metrics"""
        
        # Generate synthetic confusion matrix
        n_samples = 1000
        true_labels = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
        pred_probs = np.random.beta(2, 5, n_samples)
        predicted_labels = (pred_probs > 0.5).astype(int)
        
        # Adjust predictions to be more realistic
        for i in range(len(true_labels)):
            if true_labels[i] == 1:
                pred_probs[i] = np.random.beta(3, 2)  # Higher prob for positive class
            else:
                pred_probs[i] = np.random.beta(2, 3)  # Lower prob for negative class
        
        predicted_labels = (pred_probs > 0.5).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
        
        return {
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        }
    
    def _get_regression_metrics(self):
        """Generate regression metrics"""
        
        n_samples = 1000
        actual_values = np.random.normal(1000, 200, n_samples)
        noise_factor = 0.1
        predicted_values = actual_values + np.random.normal(0, actual_values * noise_factor)
        
        return {
            'actual_values': actual_values.tolist(),
            'predicted_values': predicted_values.tolist()
        }
    
    def get_cv_results(self, model_name):
        """Get cross-validation results"""
        
        # Generate CV scores
        n_folds = 5
        base_score = np.random.uniform(0.80, 0.92)
        cv_scores = np.random.normal(base_score, 0.02, n_folds)
        cv_scores = np.clip(cv_scores, 0.75, 0.95)
        
        # Learning curve data
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        val_scores = []
        
        for size in train_sizes:
            # Training scores (should be higher and increase with more data)
            train_score = base_score + 0.05 - 0.03 * np.exp(-size * 3)
            train_scores.append(np.random.normal(train_score, 0.01, n_folds))
            
            # Validation scores (should be lower but increase with more data)
            val_score = base_score - 0.02 + 0.02 * size
            val_scores.append(np.random.normal(val_score, 0.015, n_folds))
        
        # Hyperparameter tuning results
        hyperparameters = {
            'n_estimators': ['learning_rate', 'max_depth', 'min_samples_split'],
            'learning_rate': ['n_estimators', 'max_depth', 'subsample'],
            'max_depth': ['n_estimators', 'learning_rate', 'min_samples_leaf']
        }
        
        best_params = {
            'n_estimators': random.choice([100, 200, 300]),
            'learning_rate': random.choice([0.01, 0.1, 0.2]),
            'max_depth': random.choice([3, 5, 7, 10])
        }
        
        param_importance = {
            'n_estimators': np.random.uniform(0.20, 0.35),
            'learning_rate': np.random.uniform(0.25, 0.40),
            'max_depth': np.random.uniform(0.15, 0.30),
            'min_samples_split': np.random.uniform(0.10, 0.20)
        }
        
        return {
            'cv_scores': cv_scores.tolist(),
            'train_sizes': (train_sizes * 1000).astype(int).tolist(),
            'train_scores': np.array(train_scores).tolist(),
            'val_scores': np.array(val_scores).tolist(),
            'hyperparameter_tuning': {
                'best_params': best_params,
                'param_importance': param_importance
            }
        }
    
    def get_feature_analysis(self, model_name):
        """Get feature importance and analysis"""
        
        # Define features based on model type
        if "Sales Forecasting" in model_name:
            features = ['Seasonality', 'Historical Sales', 'Marketing Spend', 'Price', 'Economic Indicators', 'Competition']
        elif "Churn Prediction" in model_name:
            features = ['Days Since Last Purchase', 'Purchase Frequency', 'Customer Support Interactions', 
                       'Email Engagement', 'Order Value', 'Account Age']
        elif "Customer Segmentation" in model_name:
            features = ['CLV', 'Purchase Frequency', 'Average Order Value', 'Recency', 'Product Categories', 'Geography']
        else:
            features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6']
        
        # Generate feature importance (normalized to sum to 1)
        importance_values = np.random.exponential(1, len(features))
        importance_values = importance_values / importance_values.sum()
        importance = dict(zip(features, importance_values))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Generate correlation matrix
        n_features = len(features)
        correlation_matrix = np.random.uniform(-0.3, 0.8, (n_features, n_features))
        # Make it symmetric and set diagonal to 1
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = np.clip(correlation_matrix, -1, 1)
        
        # Generate feature distributions
        distributions = {}
        for feature in features:
            if 'frequency' in feature.lower():
                distributions[feature] = np.random.exponential(2, 1000)
            elif 'value' in feature.lower() or 'clv' in feature.lower():
                distributions[feature] = np.random.lognormal(5, 1, 1000)
            elif 'days' in feature.lower():
                distributions[feature] = np.random.exponential(30, 1000)
            else:
                distributions[feature] = np.random.normal(0, 1, 1000)
        
        return {
            'importance': importance,
            'correlation_matrix': correlation_matrix.tolist(),
            'distributions': distributions
        }
    
    def get_realtime_monitoring(self, model_name):
        """Get real-time monitoring data"""
        
        # Generate time series data for monitoring
        n_points = 100
        
        # Response times (should be low with occasional spikes)
        base_response_time = 50
        response_times = np.random.exponential(base_response_time, n_points)
        # Add occasional spikes
        spike_indices = np.random.choice(n_points, size=5, replace=False)
        response_times[spike_indices] *= np.random.uniform(2, 5, len(spike_indices))
        response_times = np.clip(response_times, 10, 200)
        
        # Prediction volume
        base_volume = 20
        prediction_volume = np.random.poisson(base_volume, n_points)
        # Add business hours pattern
        for i in range(n_points):
            hour_factor = 1 + 0.5 * np.sin(2 * np.pi * i / 24)  # Daily pattern
            prediction_volume[i] = int(prediction_volume[i] * hour_factor)
        
        # Data drift detection
        drift_features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        drift_scores = {feature: np.random.uniform(0, 0.8) for feature in drift_features}
        
        # Model performance history
        performance_history = []
        base_performance = 0.85
        for i in range(30):  # 30 time periods
            # Gradual degradation with some noise
            performance = base_performance - 0.001 * i + np.random.normal(0, 0.01)
            performance = np.clip(performance, 0.75, 0.95)
            performance_history.append(performance)
        
        # Generate alerts
        alerts = []
        
        # High response time alert
        if max(response_times) > 150:
            alerts.append({
                'severity': 'high',
                'message': 'Model response time exceeded 150ms threshold',
                'timestamp': (datetime.now() - timedelta(minutes=np.random.randint(1, 60))).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Data drift alert
        high_drift_features = [f for f, score in drift_scores.items() if score > 0.6]
        if high_drift_features:
            alerts.append({
                'severity': 'medium',
                'message': f'Data drift detected in features: {", ".join(high_drift_features)}',
                'timestamp': (datetime.now() - timedelta(hours=np.random.randint(1, 24))).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Performance degradation alert
        if performance_history[-1] < 0.82:
            alerts.append({
                'severity': 'medium',
                'message': 'Model performance has degraded below 82% accuracy',
                'timestamp': (datetime.now() - timedelta(hours=np.random.randint(2, 12))).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Info alerts
        alerts.append({
            'severity': 'low',
            'message': 'Model retrained successfully with updated data',
            'timestamp': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return {
            'response_times': response_times.tolist(),
            'prediction_volume': prediction_volume.tolist(),
            'data_drift': {
                'drift_scores': drift_scores,
                'performance_history': performance_history
            },
            'alerts': alerts
        }
    
    def compare_models(self, model_names):
        """Compare multiple models"""
        
        performance_metrics = {}
        resource_utilization = {}
        complexity = {}
        
        for model_name in model_names:
            # Performance metrics
            base_accuracy = np.random.uniform(0.80, 0.95)
            performance_metrics[model_name] = {
                'Accuracy': base_accuracy,
                'Precision': base_accuracy * np.random.uniform(0.95, 1.05),
                'Recall': base_accuracy * np.random.uniform(0.90, 1.02),
                'F1 Score': base_accuracy * np.random.uniform(0.92, 1.03)
            }
            
            # Resource utilization
            resource_utilization[model_name] = {
                'training_time': np.random.uniform(5, 120),  # minutes
                'memory_usage': np.random.uniform(512, 4096),  # MB
                'cpu_usage': np.random.uniform(30, 90)  # percentage
            }
            
            # Model complexity
            complexity[model_name] = {
                'Number of Features': np.random.randint(5, 50),
                'Model Parameters': np.random.randint(1000, 100000),
                'Training Data Size': np.random.randint(10000, 1000000)
            }
        
        return {
            'performance_metrics': performance_metrics,
            'resource_utilization': resource_utilization,
            'complexity': complexity
        }
    
    def calculate_business_impact(self, model_name):
        """Calculate business impact metrics"""
        
        if "Sales Forecasting" in model_name:
            return {
                'revenue_impact': np.random.uniform(100000, 500000),
                'cost_savings': np.random.uniform(50000, 200000),
                'efficiency_gain': np.random.uniform(15, 35)  # percentage
            }
        elif "Churn Prediction" in model_name:
            return {
                'customers_retained': np.random.randint(100, 500),
                'revenue_saved': np.random.uniform(200000, 800000),
                'retention_improvement': np.random.uniform(5, 20)  # percentage
            }
        elif "Customer Segmentation" in model_name:
            return {
                'marketing_efficiency': np.random.uniform(20, 40),  # percentage
                'conversion_improvement': np.random.uniform(8, 25),  # percentage
                'cost_per_acquisition_reduction': np.random.uniform(15, 30)  # percentage
            }
        else:
            return {
                'roi': np.random.uniform(120, 300),  # percentage
                'time_savings': np.random.uniform(10, 50),  # hours per week
                'accuracy_improvement': np.random.uniform(5, 20)  # percentage
            }
