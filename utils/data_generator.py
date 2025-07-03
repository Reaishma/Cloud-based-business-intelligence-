import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataGenerator:
    """Generate synthetic business data for ML workflows"""
    
    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def generate_sales_data(self, days=30):
        """Generate synthetic sales data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sales with trend and seasonality
        base_sales = 1000
        trend = np.linspace(0, 200, len(dates))
        seasonality = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
        noise = np.random.normal(0, 50, len(dates))
        
        sales = base_sales + trend + seasonality + noise
        sales = np.maximum(sales, 100)  # Ensure positive values
        
        return pd.DataFrame({
            'date': dates,
            'sales': sales
        })
    
    def generate_customer_segments(self):
        """Generate customer segment distribution"""
        return {
            'High Value': np.random.randint(200, 400),
            'Medium Value': np.random.randint(500, 800),
            'Low Value': np.random.randint(800, 1200),
            'New Customers': np.random.randint(300, 500),
            'At Risk': np.random.randint(100, 250)
        }
    
    def generate_revenue_data(self, days=90):
        """Generate revenue data with monthly aggregation"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Generate 3 months of data
        selected_months = months[:3]
        base_revenue = 100000
        growth_rate = 0.08
        
        revenue = []
        for i, month in enumerate(selected_months):
            monthly_revenue = base_revenue * (1 + growth_rate * i) + np.random.normal(0, 5000)
            revenue.append(max(monthly_revenue, 50000))
        
        return pd.DataFrame({
            'month': selected_months,
            'revenue': revenue
        })
    
    def generate_customer_data(self, n_customers=1000):
        """Generate synthetic customer data for ML models"""
        
        # Customer demographics
        ages = np.random.normal(35, 12, n_customers)
        ages = np.clip(ages, 18, 80)
        
        # Purchase behavior
        purchase_frequency = np.random.exponential(2, n_customers)
        avg_order_value = np.random.lognormal(5, 0.8, n_customers)
        
        # Customer lifetime value
        clv = purchase_frequency * avg_order_value * np.random.uniform(0.5, 3, n_customers)
        
        # Engagement metrics
        website_visits = np.random.poisson(8, n_customers)
        email_opens = np.random.binomial(10, 0.3, n_customers)
        
        # Churn indicators
        days_since_last_purchase = np.random.exponential(30, n_customers)
        support_tickets = np.random.poisson(1, n_customers)
        
        return pd.DataFrame({
            'customer_id': range(n_customers),
            'age': ages,
            'purchase_frequency': purchase_frequency,
            'avg_order_value': avg_order_value,
            'clv': clv,
            'website_visits': website_visits,
            'email_opens': email_opens,
            'days_since_last_purchase': days_since_last_purchase,
            'support_tickets': support_tickets
        })
    
    def generate_product_data(self, n_products=100):
        """Generate synthetic product data"""
        
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        
        products = []
        for i in range(n_products):
            product = {
                'product_id': f'P{i:04d}',
                'category': random.choice(categories),
                'price': np.random.uniform(10, 500),
                'cost': np.random.uniform(5, 250),
                'inventory': np.random.randint(0, 1000),
                'sales_last_30_days': np.random.randint(0, 100),
                'rating': np.random.uniform(3.0, 5.0),
                'reviews_count': np.random.randint(0, 500)
            }
            products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_time_series_data(self, start_date, end_date, freq='H'):
        """Generate time series data for various metrics"""
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_points = len(dates)
        
        # Generate multiple time series
        data = {
            'datetime': dates,
            'website_traffic': np.random.poisson(100, n_points) + 50 * np.sin(2 * np.pi * np.arange(n_points) / 24),
            'sales_volume': np.random.poisson(20, n_points) + 10 * np.sin(2 * np.pi * np.arange(n_points) / 24),
            'support_tickets': np.random.poisson(5, n_points),
            'server_load': np.random.uniform(0.3, 0.9, n_points),
            'conversion_rate': np.random.beta(2, 8, n_points)
        }
        
        return pd.DataFrame(data)
    
    def generate_satisfaction_data(self):
        """Generate customer satisfaction metrics"""
        
        # NPS score calculation
        promoters = np.random.randint(40, 60)
        detractors = np.random.randint(10, 25)
        passives = 100 - promoters - detractors
        nps_score = promoters - detractors
        
        return {
            'nps_score': nps_score,
            'promoters': promoters,
            'detractors': detractors,
            'passives': passives,
            'satisfaction_scores': {
                'product_quality': np.random.uniform(4.0, 4.8),
                'customer_service': np.random.uniform(3.8, 4.6),
                'pricing': np.random.uniform(3.5, 4.2),
                'delivery': np.random.uniform(4.1, 4.7),
                'user_experience': np.random.uniform(3.9, 4.5)
            }
        }
    
    def generate_market_data(self):
        """Generate market analysis data"""
        
        competitors = ['Competitor A', 'Competitor B', 'Competitor C', 'Competitor D']
        
        market_data = {
            'total_market_size': np.random.uniform(1.5, 2.5),  # in billions
            'growth_rate': np.random.uniform(8, 15),  # percentage
            'our_market_share': np.random.uniform(12, 18),  # percentage
            'competitor_shares': {
                comp: np.random.uniform(15, 25) for comp in competitors
            }
        }
        
        return market_data
    
    def generate_financial_data(self):
        """Generate financial metrics"""
        
        return {
            'revenue_growth': np.random.uniform(10, 25),
            'profit_margin': np.random.uniform(15, 30),
            'operating_expenses': np.random.uniform(2, 5),  # millions
            'customer_acquisition_cost': np.random.uniform(50, 150),
            'lifetime_value': np.random.uniform(500, 1500),
            'churn_rate': np.random.uniform(3, 8),
            'monthly_recurring_revenue': np.random.uniform(100, 200)  # thousands
        }
    
    def generate_operational_data(self):
        """Generate operational metrics"""
        
        return {
            'order_fulfillment_time': np.random.uniform(1.5, 3.5),  # days
            'inventory_turnover': np.random.uniform(8, 15),
            'supply_chain_efficiency': np.random.uniform(85, 95),  # percentage
            'employee_productivity': np.random.uniform(90, 110),  # index
            'system_uptime': np.random.uniform(99.0, 99.9),  # percentage
            'response_time': np.random.uniform(100, 500)  # milliseconds
      }
      
      
