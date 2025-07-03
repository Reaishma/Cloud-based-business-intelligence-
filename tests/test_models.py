"""
Unit Tests for ML Models and Data Processing
Test suite for the Cloud BI Platform components
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_processing import DataProcessor
from scripts.model_training import ModelTrainer
from utils.data_generator import DataGenerator
from utils.ml_models import MLModels
from utils.metrics import MetricsCalculator

class TestDataProcessor(unittest.TestCase):
    """Test data processing pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
    
    def test_extract_sales_data(self):
        """Test sales data extraction"""
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        result = self.processor.extract_sales_data(start_date, end_date)
        
        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            self.assertIn('transaction_id', result.columns)
            self.assertIn('customer_id', result.columns)
            self.assertIn('total_amount', result.columns)
    
    def test_transform_customer_data(self):
        """Test customer data transformation"""
        # Create sample data
        sample_data = pd.DataFrame({
            'customer_id': ['CUST001', 'CUST001', 'CUST002'],
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'transaction_date': ['2024-01-15', '2024-01-20', '2024-01-18'],
            'total_amount': [100.0, 150.0, 200.0]
        })
        
        result = self.processor.transform_customer_data(sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('customer_id', result.columns)
        self.assertIn('recency', result.columns)
        self.assertIn('frequency', result.columns)
        self.assertIn('monetary_total', result.columns)

class TestModelTrainer(unittest.TestCase):
    """Test ML model training pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = ModelTrainer()
        
        # Create sample datasets
        self.sample_customers = pd.DataFrame({
            'customer_id': [f'CUST{i:03d}' for i in range(1, 101)],
            'customer_key': range(1, 101),
            'lifetime_value': np.random.uniform(100, 2000, 100),
            'total_orders': np.random.randint(1, 20, 100),
            'registration_date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
        self.sample_sales = pd.DataFrame({
            'customer_id': np.random.choice([f'CUST{i:03d}' for i in range(1, 101)], 200),
            'transaction_id': [f'TXN{i:04d}' for i in range(1, 201)],
            'transaction_date': pd.date_range('2024-01-01', periods=200, freq='D'),
            'total_amount': np.random.uniform(10, 500, 200)
        })
    
    def test_prepare_churn_prediction_data(self):
        """Test churn prediction data preparation"""
        X, y = self.trainer.prepare_churn_prediction_data(
            self.sample_customers, self.sample_sales
        )
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], 6)  # Expected number of features
    
    def test_prepare_customer_segmentation_data(self):
        """Test customer segmentation data preparation"""
        X = self.trainer.prepare_customer_segmentation_data(
            self.sample_customers, self.sample_sales
        )
        
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape[1], 4)  # RFM + LTV features

class TestDataGenerator(unittest.TestCase):
    """Test synthetic data generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DataGenerator()
    
    def test_generate_sales_data(self):
        """Test sales data generation"""
        result = self.generator.generate_sales_data(30)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 31)  # 30 days + 1
        self.assertIn('date', result.columns)
        self.assertIn('sales', result.columns)
        self.assertTrue(all(result['sales'] > 0))
    
    def test_generate_customer_segments(self):
        """Test customer segment generation"""
        result = self.generator.generate_customer_segments()
        
        self.assertIsInstance(result, dict)
        self.assertIn('High Value', result)
        self.assertIn('Medium Value', result)
        self.assertIn('Low Value', result)
        self.assertTrue(all(isinstance(v, (int, np.integer)) for v in result.values()))
    
    def test_generate_customer_data(self):
        """Test customer data generation"""
        result = self.generator.generate_customer_data(100)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 100)
        self.assertIn('customer_id', result.columns)
        self.assertIn('clv', result.columns)
        self.assertTrue(all(result['clv'] > 0))

class TestMLModels(unittest.TestCase):
    """Test ML model simulation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ml_models = MLModels()
    
    def test_generate_sales_forecast(self):
        """Test sales forecasting"""
        result = self.ml_models.generate_sales_forecast(30)
        
        self.assertIsInstance(result, dict)
        self.assertIn('historical', result)
        self.assertIn('forecast', result)
        
        historical = result['historical']
        forecast = result['forecast']
        
        self.assertIn('date', historical)
        self.assertIn('sales', historical)
        self.assertEqual(len(forecast['date']), 30)
        self.assertEqual(len(forecast['sales']), 30)
    
    def test_predict_customer_ltv(self):
        """Test customer LTV prediction"""
        result = self.ml_models.predict_customer_ltv(100)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 100)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result))
        self.assertTrue(all(x > 0 for x in result))
    
    def test_run_sales_pipeline(self):
        """Test sales pipeline execution"""
        result = self.ml_models.run_sales_pipeline(30, "ARIMA + ML", 0.8)
        
        self.assertIsInstance(result, dict)
        self.assertIn('mae', result)
        self.assertIn('rmse', result)
        self.assertIn('r2', result)
        self.assertTrue(result['mae'] > 0)
        self.assertTrue(result['rmse'] > 0)

class TestMetricsCalculator(unittest.TestCase):
    """Test metrics calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = MetricsCalculator()
    
    def test_get_model_performance(self):
        """Test model performance metrics"""
        result = self.metrics.get_model_performance("Sales Forecasting Model")
        
        self.assertIsInstance(result, dict)
        self.assertIn('accuracy', result)
        self.assertIn('precision', result)
        self.assertIn('recall', result)
        self.assertIn('f1_score', result)
        
        # Check metric ranges
        self.assertGreaterEqual(result['accuracy'], 0)
        self.assertLessEqual(result['accuracy'], 1)
    
    def test_get_detailed_metrics(self):
        """Test detailed metrics generation"""
        result = self.metrics.get_detailed_metrics("Churn Prediction Model")
        
        self.assertIsInstance(result, dict)
        self.assertIn('confusion_matrix', result)
        self.assertIn('roc_curve', result)
        
        roc_curve = result['roc_curve']
        self.assertIn('fpr', roc_curve)
        self.assertIn('tpr', roc_curve)
        self.assertIn('auc', roc_curve)
    
    def test_get_feature_analysis(self):
        """Test feature analysis"""
        result = self.metrics.get_feature_analysis("Sales Forecasting Model")
        
        self.assertIsInstance(result, dict)
        self.assertIn('importance', result)
        self.assertIn('correlation_matrix', result)
        self.assertIn('distributions', result)
        
        importance = result['importance']
        self.assertTrue(all(isinstance(v, (int, float)) for v in importance.values()))

class TestDataIntegration(unittest.TestCase):
    """Test data integration and end-to-end workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DataGenerator()
        self.processor = DataProcessor()
        self.ml_models = MLModels()
    
    def test_end_to_end_pipeline(self):
        """Test complete data pipeline"""
        # Generate data
        sales_data = self.generator.generate_sales_data(30)
        customer_data = self.generator.generate_customer_data(50)
        
        # Verify data integrity
        self.assertFalse(sales_data.empty)
        self.assertFalse(customer_data.empty)
        
        # Test ML model operations
        forecast = self.ml_models.generate_sales_forecast(7)
        self.assertIn('forecast', forecast)
        
        clv_predictions = self.ml_models.predict_customer_ltv(50)
        self.assertEqual(len(clv_predictions), 50)
    
    def test_data_consistency(self):
        """Test data consistency across components"""
        # Generate consistent datasets
        sales_data = self.generator.generate_sales_data(30)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(sales_data['date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(sales_data['sales']))
        
        # Check data ranges
        self.assertTrue(all(sales_data['sales'] >= 0))

class TestModelValidation(unittest.TestCase):
    """Test model validation and performance monitoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = MetricsCalculator()
    
    def test_cross_validation_results(self):
        """Test cross-validation metrics"""
        result = self.metrics.get_cv_results("Churn Prediction Model")
        
        self.assertIsInstance(result, dict)
        self.assertIn('cv_scores', result)
        self.assertIn('train_sizes', result)
        self.assertIn('hyperparameter_tuning', result)
        
        cv_scores = result['cv_scores']
        self.assertTrue(all(0 <= score <= 1 for score in cv_scores))
    
    def test_realtime_monitoring(self):
        """Test real-time monitoring data"""
        result = self.metrics.get_realtime_monitoring("Sales Forecasting Model")
        
        self.assertIsInstance(result, dict)
        self.assertIn('response_times', result)
        self.assertIn('prediction_volume', result)
        self.assertIn('data_drift', result)
        self.assertIn('alerts', result)
        
        response_times = result['response_times']
        self.assertTrue(all(t > 0 for t in response_times))

def run_all_tests():
    """Run all test suites"""
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataProcessor,
        TestModelTrainer,
        TestDataGenerator,
        TestMLModels,
        TestMetricsCalculator,
        TestDataIntegration,
        TestModelValidation
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check output above for details.")
        sys.exit(1)
      
