"""
Data Processing Pipeline for Cloud BI Platform
Simulates AWS Glue / Google Cloud Dataflow ETL processes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import boto3
from google.cloud import bigquery
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class for BI platform"""
    
    def __init__(self, config_path: str = "config/bi_platform_config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_cloud_connections()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_cloud_connections(self):
        """Initialize cloud service connections"""
        try:
            # AWS connections
            self.s3_client = boto3.client('s3')
            self.redshift_client = boto3.client('redshift-data')
            
            # GCP connections
            self.bigquery_client = bigquery.Client()
            
            logger.info("Cloud connections initialized successfully")
        except Exception as e:
            logger.warning(f"Cloud connections not available: {e}")
    
    def extract_sales_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract sales data from multiple sources"""
        logger.info(f"Extracting sales data from {start_date} to {end_date}")
        
        # Simulate data extraction from various sources
        query = f"""
        SELECT 
            t.transaction_id,
            t.customer_id,
            t.product_id,
            t.quantity,
            t.unit_price,
            t.total_amount,
            t.transaction_date,
            t.region,
            c.customer_segment,
            c.lifetime_value,
            p.category,
            p.brand
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        JOIN products p ON t.product_id = p.product_id
        WHERE t.transaction_date BETWEEN '{start_date}' AND '{end_date}'
        """
        
        # In a real implementation, this would execute against the actual database
        # For simulation, we'll load from our CSV files
        try:
            sales_df = pd.read_csv('data/raw/sales_data.csv')
            customers_df = pd.read_csv('data/raw/customer_data.csv')
            products_df = pd.read_csv('data/raw/product_data.csv')
            
            # Merge data to simulate the join operation
            merged_df = sales_df.merge(customers_df, on='customer_id', how='left')
            merged_df = merged_df.merge(products_df[['product_id', 'category', 'brand']], 
                                      on='product_id', how='left')
            
            logger.info(f"Extracted {len(merged_df)} sales records")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error extracting sales data: {e}")
            return pd.DataFrame()
    
    def transform_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform customer data for ML model training"""
        logger.info("Transforming customer data")
        
        # Calculate RFM metrics
        current_date = datetime.now()
        
        customer_features = df.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - pd.to_datetime(x).max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'total_amount': ['sum', 'mean']  # Monetary
        }).round(2)
        
        customer_features.columns = ['recency', 'frequency', 'monetary_total', 'monetary_avg']
        
        # Add derived features
        customer_features['clv_score'] = (
            customer_features['frequency'] * customer_features['monetary_avg'] * 0.1
        )
        
        # Categorize customers
        customer_features['value_segment'] = pd.cut(
            customer_features['monetary_total'],
            bins=[0, 500, 1500, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        logger.info(f"Transformed {len(customer_features)} customer records")
        return customer_features.reset_index()
    
    def create_feature_store(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create feature store for ML models"""
        logger.info("Creating feature store")
        
        features = {
            'sales_features': {
                'daily_sales': df.groupby('transaction_date')['total_amount'].sum().to_dict(),
                'product_performance': df.groupby('product_id')['total_amount'].sum().to_dict(),
                'regional_metrics': df.groupby('region')['total_amount'].agg(['sum', 'mean']).to_dict()
            },
            'customer_features': {
                'segment_distribution': df['customer_segment'].value_counts().to_dict(),
                'ltv_stats': df.groupby('customer_segment')['lifetime_value'].agg(['mean', 'std']).to_dict()
            },
            'temporal_features': {
                'seasonality': self._calculate_seasonality(df),
                'trends': self._calculate_trends(df)
            }
        }
        
        # Save to feature store
        with open('data/features/feature_store.json', 'w') as f:
            json.dump(features, f, indent=2, default=str)
        
        logger.info("Feature store created successfully")
        return features
    
    def _calculate_seasonality(self, df: pd.DataFrame) -> Dict:
        """Calculate seasonal patterns in sales data"""
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['month'] = df['transaction_date'].dt.month
        df['day_of_week'] = df['transaction_date'].dt.day_of_week
        
        monthly_sales = df.groupby('month')['total_amount'].mean().to_dict()
        weekly_sales = df.groupby('day_of_week')['total_amount'].mean().to_dict()
        
        return {
            'monthly_pattern': monthly_sales,
            'weekly_pattern': weekly_sales
        }
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate trend indicators"""
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        daily_sales = df.groupby('transaction_date')['total_amount'].sum().sort_index()
        
        # Calculate growth rate
        growth_rate = daily_sales.pct_change().mean()
        
        # Calculate moving averages
        ma_7 = daily_sales.rolling(window=7).mean().iloc[-1]
        ma_30 = daily_sales.rolling(window=30).mean().iloc[-1]
        
        return {
            'growth_rate': growth_rate,
            'ma_7': ma_7,
            'ma_30': ma_30,
            'trend_direction': 'upward' if ma_7 > ma_30 else 'downward'
        }
    
    def load_to_warehouse(self, df: pd.DataFrame, table_name: str):
        """Load processed data to data warehouse"""
        logger.info(f"Loading data to warehouse table: {table_name}")
        
        # Save locally (in production, this would go to Redshift/BigQuery)
        output_path = f'data/processed/{table_name}.parquet'
        df.to_parquet(output_path, index=False)
        
        # Simulate warehouse loading
        warehouse_config = self.config['data_sources']['warehouse']
        logger.info(f"Data loaded to {warehouse_config['type']} warehouse")
        
        return output_path
    
    def run_etl_pipeline(self):
        """Execute the complete ETL pipeline"""
        logger.info("Starting ETL pipeline execution")
        
        try:
            # Extract
            start_date = "2024-01-01"
            end_date = "2024-01-31"
            raw_data = self.extract_sales_data(start_date, end_date)
            
            if raw_data.empty:
                logger.error("No data extracted, stopping pipeline")
                return False
            
            # Transform
            customer_features = self.transform_customer_data(raw_data)
            feature_store = self.create_feature_store(raw_data)
            
            # Load
            self.load_to_warehouse(raw_data, 'sales_fact')
            self.load_to_warehouse(customer_features, 'customer_features')
            
            logger.info("ETL pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            return False

def main():
    """Main execution function"""
    processor = DataProcessor()
    success = processor.run_etl_pipeline()
    
    if success:
        print("✅ Data processing pipeline completed successfully")
    else:
        print("❌ Data processing pipeline failed")

if __name__ == "__main__":
    main()
  
