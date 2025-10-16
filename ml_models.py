import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import random

class MLModels:
    """Simulate ML models and workflows for the BI platform"""
    
    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.scaler = StandardScaler()
    
    def generate_sales_forecast(self, days=30):
        """Generate sales forecast data with historical and predicted values"""
        
        # Historical data (last 60 days)
        historical_dates = pd.date_range(start=datetime.now() - timedelta(days=60), 
                                       end=datetime.now(), freq='D')
        historical_sales = 1000 + 200 * np.sin(2 * np.pi * np.arange(len(historical_dates)) / 7) + \
                          np.random.normal(0, 100, len(historical_dates))
        historical_sales = np.maximum(historical_sales, 200)
        
        # Forecast data (next 30 days)
        forecast_dates = pd.date_range(start=datetime.now() + timedelta(days=1), 
                                     periods=days, freq='D')
        
        # Generate forecast with trend and uncertainty
        trend = np.linspace(0, 150, days)
        seasonal = 200 * np.sin(2 * np.pi * np.arange(days) / 7)
        forecast_sales = 1100 + trend + seasonal + np.random.normal(0, 50, days)
        forecast_sales = np.maximum(forecast_sales, 300)
        
        # Confidence intervals
        uncertainty = np.random.uniform(0.1, 0.2, days)
        upper_bound = forecast_sales * (1 + uncertainty)
        lower_bound = forecast_sales * (1 - uncertainty)
        
        return {
            'historical': {
                'date': historical_dates.tolist(),
                'sales': historical_sales.tolist()
            },
            'forecast': {
                'date': forecast_dates.tolist(),
                'sales': forecast_sales.tolist(),
                'upper_bound': upper_bound.tolist(),
                'lower_bound': lower_bound.tolist()
            }
        }
    
    def predict_customer_ltv(self, n_customers=1000):
        """Predict customer lifetime value distribution"""
        
        # Generate realistic CLV distribution
        base_clv = np.random.lognormal(mean=6.5, sigma=0.8, size=n_customers)
        base_clv = np.clip(base_clv, 100, 5000)
        
        return base_clv.tolist()
    
    def predict_churn_risk(self):
        """Predict customer churn risk distribution"""
        
        total_customers = np.random.randint(1800, 2200)
        
        # Risk distribution
        high_risk = int(total_customers * np.random.uniform(0.12, 0.18))
        medium_risk = int(total_customers * np.random.uniform(0.25, 0.35))
        low_risk = total_customers - high_risk - medium_risk
        
        return {
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': max(low_risk, 0),
            'total_customers': total_customers
        }
    
    def run_sales_pipeline(self, horizon, model_type, split):
        """Simulate running a sales forecasting pipeline"""
        
        # Generate synthetic training results
        n_samples = 1000
        
        # Create synthetic actual vs predicted data
        actual_values = np.random.normal(1000, 200, n_samples)
        noise_factor = np.random.uniform(0.05, 0.15)
        predicted_values = actual_values + np.random.normal(0, actual_values * noise_factor)
        
        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        r2 = r2_score(actual_values, predicted_values)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'mae_delta': np.random.uniform(-2, 2),
            'rmse_delta': np.random.uniform(-2, 2),
            'mape_delta': np.random.uniform(-1, 1),
            'r2_delta': np.random.uniform(-0.05, 0.05),
            'actual_values': actual_values.tolist(),
            'predicted_values': predicted_values.tolist()
        }
    
    def generate_customer_segments(self, n_clusters=5):
        """Generate customer segmentation data"""
        
        n_customers = 1000
        
        # Generate synthetic customer features
        features = np.random.randn(n_customers, 3)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        segments = kmeans.fit_predict(features)
        
        return pd.DataFrame({
            'feature_1': features[:, 0],
            'feature_2': features[:, 1],
            'feature_3': features[:, 2],
            'segment': segments
        })
    
    def run_segmentation_pipeline(self, n_clusters, features, algorithm):
        """Simulate customer segmentation pipeline execution"""
        
        # Generate synthetic clustering results
        silhouette_score = np.random.uniform(0.3, 0.7)
        inertia = np.random.uniform(500, 1500)
        calinski_score = np.random.uniform(50, 200)
        
        # Generate segment characteristics
        segment_chars = []
        for i in range(n_clusters):
            segment_chars.append({
                'segment': i,
                'size': np.random.randint(50, 300),
                'avg_revenue': np.random.uniform(500, 2000),
                'avg_frequency': np.random.uniform(2, 12),
                'churn_rate': np.random.uniform(0.05, 0.25)
            })
        
        return {
            'silhouette_score': silhouette_score,
            'inertia': inertia,
            'calinski_score': calinski_score,
            'segment_characteristics': segment_chars
        }
    
    def analyze_churn_risk(self):
        """Analyze customer churn risk factors"""
        
        total_customers = 2000
        high_risk = int(total_customers * 0.15)
        medium_risk = int(total_customers * 0.30)
        low_risk = total_customers - high_risk - medium_risk
        
        return {
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk
        }
    
    def run_churn_pipeline(self, algorithm, window, threshold):
        """Execute churn prediction pipeline"""
        
        # Simulate model performance
        accuracy = np.random.uniform(0.85, 0.95)
        precision = np.random.uniform(0.82, 0.92)
        recall = np.random.uniform(0.80, 0.90)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Feature importance
        features = {
            'Days Since Last Purchase': np.random.uniform(0.20, 0.30),
            'Purchase Frequency': np.random.uniform(0.15, 0.25),
            'Average Order Value': np.random.uniform(0.10, 0.20),
            'Customer Support Interactions': np.random.uniform(0.08, 0.15),
            'Email Engagement': np.random.uniform(0.06, 0.12),
            'Website Activity': np.random.uniform(0.05, 0.10)
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'feature_importance': features
        }
    
    def run_price_optimization(self, objective, constraints, category):
        """Execute price optimization pipeline"""
        
        current_price = np.random.uniform(20, 100)
        optimal_price = current_price * np.random.uniform(0.9, 1.2)
        
        price_change = (optimal_price - current_price) / current_price
        
        expected_revenue = np.random.uniform(50000, 150000)
        revenue_change = np.random.uniform(0.05, 0.25)
        
        profit_margin = np.random.uniform(0.15, 0.35)
        margin_change = np.random.uniform(-0.02, 0.08)
        
        return {
            'optimal_price': optimal_price,
            'price_change': price_change,
            'expected_revenue': expected_revenue,
            'revenue_change': revenue_change,
            'profit_margin': profit_margin,
            'margin_change': margin_change
        }
    
    def generate_revenue_forecast(self):
        """Generate comprehensive revenue forecast"""
        
        # Historical data (6 months)
        historical_dates = pd.date_range(start=datetime.now() - timedelta(days=180), 
                                       end=datetime.now(), freq='M')
        base_revenue = 100000
        growth = np.linspace(0, 50000, len(historical_dates))
        seasonal = 20000 * np.sin(2 * np.pi * np.arange(len(historical_dates)) / 12)
        historical_revenue = base_revenue + growth + seasonal + np.random.normal(0, 5000, len(historical_dates))
        
        # Forecast (6 months ahead)
        forecast_dates = pd.date_range(start=datetime.now() + timedelta(days=30), 
                                     periods=6, freq='M')
        forecast_growth = np.linspace(55000, 80000, len(forecast_dates))
        forecast_seasonal = 20000 * np.sin(2 * np.pi * np.arange(len(forecast_dates)) / 12)
        forecast_revenue = base_revenue + forecast_growth + forecast_seasonal + np.random.normal(0, 3000, len(forecast_dates))
        
        # Confidence intervals
        uncertainty = 0.15
        upper_bound = forecast_revenue * (1 + uncertainty)
        lower_bound = forecast_revenue * (1 - uncertainty)
        
        return {
            'historical': {
                'dates': historical_dates.tolist(),
                'revenue': historical_revenue.tolist()
            },
            'forecast': {
                'dates': forecast_dates.tolist(),
                'revenue': forecast_revenue.tolist(),
                'upper_bound': upper_bound.tolist(),
                'lower_bound': lower_bound.tolist()
            }
        }
    
    def analyze_customer_segments(self):
        """Analyze customer segments with business metrics"""
        
        return {
            'champions': {
                'count': np.random.randint(150, 250),
                'avg_clv': np.random.uniform(2000, 3500),
                'retention': np.random.uniform(0.90, 0.98)
            },
            'potential_loyalists': {
                'count': np.random.randint(300, 500),
                'avg_clv': np.random.uniform(1000, 2000),
                'retention': np.random.uniform(0.75, 0.85)
            },
            'at_risk': {
                'count': np.random.randint(200, 350),
                'avg_clv': np.random.uniform(1500, 2500),
                'retention': np.random.uniform(0.40, 0.60)
            }
        }
    
    def generate_churn_insights(self):
        """Generate churn prediction insights"""
        
        risk_factors = {
            'Days Since Last Purchase': np.random.uniform(0.25, 0.35),
            'Declining Purchase Frequency': np.random.uniform(0.20, 0.30),
            'Reduced Order Value': np.random.uniform(0.15, 0.25),
            'Poor Customer Service Experience': np.random.uniform(0.10, 0.20),
            'Low Email Engagement': np.random.uniform(0.08, 0.15),
            'Competitor Activity': np.random.uniform(0.05, 0.12)
        }
        
        return {
            'risk_factors': risk_factors
        }
    
    def analyze_geographic_opportunities(self):
        """Analyze geographic expansion opportunities"""
        
        high_potential = [
            {
                'region': 'Southeast Asia',
                'size': np.random.randint(200, 400),
                'growth': np.random.uniform(15, 25),
                'competition': 'Low'
            },
            {
                'region': 'Eastern Europe',
                'size': np.random.randint(150, 300),
                'growth': np.random.uniform(12, 20),
                'competition': 'Medium'
            }
        ]
        
        medium_potential = [
            {
                'region': 'Latin America',
                'size': np.random.randint(100, 250),
                'growth': np.random.uniform(8, 15),
                'competition': 'Medium'
            },
            {
                'region': 'Middle East',
                'size': np.random.randint(80, 200),
                'growth': np.random.uniform(10, 18),
                'competition': 'High'
            }
        ]
        
        return {
            'high_potential': high_potential,
            'medium_potential': medium_potential
        }
    
    def identify_product_opportunities(self):
        """Identify product development opportunities"""
        
        innovations = [
            {
                'title': 'AI-Powered Personalization Engine',
                'description': 'Machine learning system for personalized product recommendations',
                'demand': 'High',
                'effort': 'Medium',
                'roi': '180%'
            },
            {
                'title': 'Mobile App Enhancement',
                'description': 'Enhanced mobile experience with AR features',
                'demand': 'Medium',
                'effort': 'High',
                'roi': '145%'
            },
            {
                'title': 'Subscription Service Model',
                'description': 'Monthly subscription box for recurring revenue',
                'demand': 'High',
                'effort': 'Low',
                'roi': '220%'
            }
        ]
        
        return {
            'innovations': innovations
        }
