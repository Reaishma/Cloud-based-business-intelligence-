"""
ML Model Training Pipeline for Cloud BI Platform
Simulates Amazon SageMaker training workflows
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
import boto3
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """ML Model Training orchestrator for BI platform"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.setup_sagemaker_session()
    
    def setup_sagemaker_session(self):
        """Initialize SageMaker session (simulated)"""
        try:
            self.sagemaker_session = boto3.Session()
            logger.info("SageMaker session initialized")
        except Exception as e:
            logger.warning(f"SageMaker not available: {e}")
    
    def load_training_data(self) -> Dict[str, pd.DataFrame]:
        """Load preprocessed training data"""
        logger.info("Loading training datasets")
        
        datasets = {}
        try:
            # Load sales data for forecasting
            datasets['sales'] = pd.read_csv('data/raw/sales_data.csv')
            datasets['customers'] = pd.read_csv('data/raw/customer_data.csv')
            datasets['products'] = pd.read_csv('data/raw/product_data.csv')
            
            logger.info(f"Loaded {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return {}
    
    def prepare_churn_prediction_data(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for churn prediction model"""
        logger.info("Preparing churn prediction dataset")
        
        # Merge customer and sales data
        customer_metrics = sales_df.groupby('customer_id').agg({
            'transaction_date': 'max',
            'transaction_id': 'count',
            'total_amount': ['sum', 'mean']
        }).round(2)
        
        customer_metrics.columns = ['last_purchase', 'purchase_count', 'total_spent', 'avg_order_value']
        
        # Calculate days since last purchase
        customer_metrics['last_purchase'] = pd.to_datetime(customer_metrics['last_purchase'])
        current_date = datetime.now()
        customer_metrics['days_since_last_purchase'] = (
            current_date - customer_metrics['last_purchase']
        ).dt.days
        
        # Merge with customer data
        merged_df = customers_df.merge(customer_metrics, on='customer_id', how='inner')
        
        # Create churn label (customers who haven't purchased in 90+ days)
        merged_df['is_churned'] = (merged_df['days_since_last_purchase'] > 90).astype(int)
        
        # Select features
        feature_columns = [
            'lifetime_value', 'total_orders', 'purchase_count', 
            'total_spent', 'avg_order_value', 'days_since_last_purchase'
        ]
        
        X = merged_df[feature_columns].fillna(0)
        y = merged_df['is_churned']
        
        logger.info(f"Prepared churn dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Churn rate: {y.mean():.2%}")
        
        return X.values, y.values
    
    def prepare_customer_segmentation_data(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> np.ndarray:
        """Prepare data for customer segmentation"""
        logger.info("Preparing customer segmentation dataset")
        
        # Calculate RFM metrics
        current_date = datetime.now()
        
        rfm_data = sales_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - pd.to_datetime(x).max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).round(2)
        
        rfm_data.columns = ['recency', 'frequency', 'monetary']
        
        # Add customer lifetime value
        customers_subset = customers_df.set_index('customer_id')[['lifetime_value']]
        rfm_data = rfm_data.merge(customers_subset, left_index=True, right_index=True, how='inner')
        
        logger.info(f"Prepared segmentation dataset: {rfm_data.shape[0]} customers, {rfm_data.shape[1]} features")
        
        return rfm_data.values
    
    def train_churn_prediction_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train churn prediction model"""
        logger.info("Training churn prediction model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10]
        }
        
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            rf_classifier, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Evaluate model
        train_score = best_model.score(X_train_scaled, y_train)
        test_score = best_model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1')
        
        # Store model and scaler
        self.models['churn_prediction'] = best_model
        self.scalers['churn_prediction'] = scaler
        
        # Save model artifacts
        model_artifacts = {
            'model': best_model,
            'scaler': scaler,
            'feature_names': ['lifetime_value', 'total_orders', 'purchase_count', 
                            'total_spent', 'avg_order_value', 'days_since_last_purchase'],
            'best_params': grid_search.best_params_,
            'performance': {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std()
            }
        }
        
        # Save to disk
        joblib.dump(model_artifacts, 'data/models/churn_prediction_model.pkl')
        
        logger.info(f"Churn model trained - Test Accuracy: {test_score:.3f}, CV F1: {cv_scores.mean():.3f}")
        
        return model_artifacts
    
    def train_customer_segmentation_model(self, X: np.ndarray) -> Dict[str, Any]:
        """Train customer segmentation model"""
        logger.info("Training customer segmentation model")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        silhouette_scores = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Select best K
        best_k = K_range[np.argmax(silhouette_scores)]
        
        # Train final model
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster statistics
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Store model and scaler
        self.models['customer_segmentation'] = kmeans
        self.scalers['customer_segmentation'] = scaler
        
        # Save model artifacts
        model_artifacts = {
            'model': kmeans,
            'scaler': scaler,
            'n_clusters': best_k,
            'cluster_centers': cluster_centers,
            'feature_names': ['recency', 'frequency', 'monetary', 'lifetime_value'],
            'performance': {
                'silhouette_score': silhouette_scores[best_k - 2],
                'inertia': kmeans.inertia_
            },
            'cluster_analysis': self._analyze_clusters(X, cluster_labels)
        }
        
        # Save to disk
        joblib.dump(model_artifacts, 'data/models/customer_segmentation_model.pkl')
        
        logger.info(f"Segmentation model trained - {best_k} clusters, Silhouette: {silhouette_scores[best_k - 2]:.3f}")
        
        return model_artifacts
    
    def _analyze_clusters(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        df = pd.DataFrame(X, columns=['recency', 'frequency', 'monetary', 'lifetime_value'])
        df['cluster'] = labels
        
        cluster_stats = df.groupby('cluster').agg({
            'recency': ['mean', 'std'],
            'frequency': ['mean', 'std'],
            'monetary': ['mean', 'std'],
            'lifetime_value': ['mean', 'std']
        }).round(2)
        
        # Assign cluster names based on characteristics
        cluster_names = {}
        for cluster_id in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster_id]
            avg_monetary = cluster_data['monetary'].mean()
            avg_frequency = cluster_data['frequency'].mean()
            avg_recency = cluster_data['recency'].mean()
            
            if avg_monetary > df['monetary'].quantile(0.75) and avg_frequency > df['frequency'].median():
                cluster_names[cluster_id] = "Champions"
            elif avg_monetary > df['monetary'].median() and avg_recency < df['recency'].median():
                cluster_names[cluster_id] = "Loyal Customers"
            elif avg_recency > df['recency'].quantile(0.75):
                cluster_names[cluster_id] = "At Risk"
            elif avg_monetary < df['monetary'].quantile(0.25):
                cluster_names[cluster_id] = "Price Sensitive"
            else:
                cluster_names[cluster_id] = f"Segment {cluster_id}"
        
        return {
            'cluster_stats': cluster_stats.to_dict(),
            'cluster_names': cluster_names,
            'cluster_sizes': df['cluster'].value_counts().to_dict()
        }
    
    def deploy_models_to_sagemaker(self):
        """Deploy trained models to SageMaker endpoints (simulated)"""
        logger.info("Deploying models to SageMaker endpoints")
        
        deployment_config = {
            'churn_prediction': {
                'endpoint_name': 'bi-platform-churn-prediction',
                'instance_type': 'ml.t2.medium',
                'initial_instance_count': 1
            },
            'customer_segmentation': {
                'endpoint_name': 'bi-platform-customer-segmentation',
                'instance_type': 'ml.t2.medium',
                'initial_instance_count': 1
            }
        }
        
        # Save deployment configuration
        with open('data/models/deployment_config.json', 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        logger.info("Models deployed successfully")
        return deployment_config
    
    def run_training_pipeline(self):
        """Execute the complete model training pipeline"""
        logger.info("Starting ML training pipeline")
        
        try:
            # Load data
            datasets = self.load_training_data()
            if not datasets:
                logger.error("No training data available")
                return False
            
            # Train churn prediction model
            X_churn, y_churn = self.prepare_churn_prediction_data(
                datasets['customers'], datasets['sales']
            )
            churn_model = self.train_churn_prediction_model(X_churn, y_churn)
            
            # Train customer segmentation model
            X_segment = self.prepare_customer_segmentation_data(
                datasets['customers'], datasets['sales']
            )
            segment_model = self.train_customer_segmentation_model(X_segment)
            
            # Deploy models
            deployment_config = self.deploy_models_to_sagemaker()
            
            # Create training summary
            training_summary = {
                'timestamp': datetime.now().isoformat(),
                'models_trained': ['churn_prediction', 'customer_segmentation'],
                'churn_model_performance': churn_model['performance'],
                'segmentation_model_performance': segment_model['performance'],
                'deployment_status': 'success'
            }
            
            # Save training summary
            with open('data/models/training_summary.json', 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            logger.info("ML training pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False

def main():
    """Main execution function"""
    trainer = ModelTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("✅ ML model training pipeline completed successfully")
    else:
        print("❌ ML model training pipeline failed")

      if __name__ == "__main__":
    main()
