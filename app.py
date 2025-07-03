import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from utils.data_generator import DataGenerator
from utils.ml_models import MLModels
from utils.metrics import MetricsCalculator

# Page configuration
st.set_page_config(
    page_title="Cloud BI Platform - ML Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'metrics_calc' not in st.session_state:
    st.session_state.metrics_calc = MetricsCalculator()

def main():
    # Sidebar
    st.sidebar.title("🌟 Cloud BI Platform")
    st.sidebar.markdown("### Amazon SageMaker Simulation")
    st.sidebar.markdown("---")
    
    # Main dashboard
    st.title("📊 Cloud Business Intelligence Platform")
    st.markdown("### AI-Powered Analytics & Predictive Insights")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate sample metrics
    total_models = 12
    active_pipelines = 8
    predictions_today = np.random.randint(1500, 3000)
    accuracy_avg = np.random.uniform(0.85, 0.95)
    
    with col1:
        st.metric(
            label="🤖 ML Models Deployed",
            value=total_models,
            delta=2
        )
    
    with col2:
        st.metric(
            label="🔄 Active Pipelines",
            value=active_pipelines,
            delta=1
        )
    
    with col3:
        st.metric(
            label="📈 Predictions Today",
            value=f"{predictions_today:,}",
            delta=f"+{np.random.randint(50, 200)}"
        )
    
    with col4:
        st.metric(
            label="🎯 Avg Model Accuracy",
            value=f"{accuracy_avg:.2%}",
            delta=f"+{np.random.uniform(0.01, 0.03):.1%}"
        )
    
    st.markdown("---")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Real-time Dashboard", "🔮 Predictive Analytics", "📋 Model Overview"])
    
    with tab1:
        show_realtime_dashboard()
    
    with tab2:
        show_predictive_analytics()
    
    with tab3:
        show_model_overview()

def show_realtime_dashboard():
    """Display real-time analytics dashboard"""
    st.subheader("Real-time Business Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales performance chart
        sales_data = st.session_state.data_generator.generate_sales_data(30)
        fig_sales = px.line(
            sales_data, 
            x='date', 
            y='sales',
            title='Daily Sales Performance (Last 30 Days)',
            color_discrete_sequence=['#FF6B6B']
        )
        fig_sales.update_layout(height=400)
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col2:
        # Customer segments pie chart
        segments = st.session_state.data_generator.generate_customer_segments()
        fig_segments = px.pie(
            values=segments.values(),
            names=segments.keys(),
            title='Customer Segments Distribution'
        )
        fig_segments.update_layout(height=400)
        st.plotly_chart(fig_segments, use_container_width=True)
    
    # Revenue trends
    st.subheader("Revenue Analytics")
    revenue_data = st.session_state.data_generator.generate_revenue_data(90)
    
    fig_revenue = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Revenue Trend', 'Revenue by Product Category'),
        vertical_spacing=0.12
    )
    
    # Monthly trend
    fig_revenue.add_trace(
        go.Scatter(
            x=revenue_data['month'],
            y=revenue_data['revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#4ECDC4', width=3)
        ),
        row=1, col=1
    )
    
    # Category breakdown
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    cat_revenue = [np.random.randint(50000, 150000) for _ in categories]
    
    fig_revenue.add_trace(
        go.Bar(
            x=categories,
            y=cat_revenue,
            name='Category Revenue',
            marker_color='#45B7D1'
        ),
        row=2, col=1
    )
    
    fig_revenue.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_revenue, use_container_width=True)

def show_predictive_analytics():
    """Display predictive analytics section"""
    st.subheader("🔮 AI-Powered Predictions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Sales Forecasting")
        
        # Generate forecast data
        forecast_data = st.session_state.ml_models.generate_sales_forecast(30)
        
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_data['historical']['date'],
                y=forecast_data['historical']['sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='#FF6B6B', width=2)
            )
        )
        
        # Forecast
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_data['forecast']['date'],
                y=forecast_data['forecast']['sales'],
                mode='lines',
                name='Predicted Sales',
                line=dict(color='#4ECDC4', width=2, dash='dash')
            )
        )
        
        # Confidence interval
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_data['forecast']['date'],
                y=forecast_data['forecast']['upper_bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            )
        )
        
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_data['forecast']['date'],
                y=forecast_data['forecast']['lower_bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                fillcolor='rgba(78, 205, 196, 0.2)'
            )
        )
        
        fig_forecast.update_layout(
            title='30-Day Sales Forecast',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast metrics
        st.markdown("**Forecast Metrics:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted Growth", "+12.5%", "2.3%")
        with col_b:
            st.metric("Confidence Level", "94.2%", "1.1%")
    
    with col2:
        st.markdown("#### Customer Lifetime Value Prediction")
        
        # CLV prediction
        clv_data = st.session_state.ml_models.predict_customer_ltv()
        
        fig_clv = px.histogram(
            x=clv_data,
            nbins=20,
            title='Customer Lifetime Value Distribution',
            color_discrete_sequence=['#45B7D1']
        )
        fig_clv.update_layout(
            xaxis_title='Predicted CLV ($)',
            yaxis_title='Number of Customers',
            height=400
        )
        st.plotly_chart(fig_clv, use_container_width=True)
        
        # CLV segments
        st.markdown("**CLV Segments:**")
        segments = {
            'High Value (>$1000)': len([x for x in clv_data if x > 1000]),
            'Medium Value ($500-$1000)': len([x for x in clv_data if 500 <= x <= 1000]),
            'Low Value (<$500)': len([x for x in clv_data if x < 500])
        }
        
        for segment, count in segments.items():
            st.write(f"• {segment}: {count} customers")
    
    # Churn prediction
    st.markdown("---")
    st.subheader("🚨 Churn Risk Analysis")
    
    churn_data = st.session_state.ml_models.predict_churn_risk()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "High Risk Customers",
            churn_data['high_risk'],
            f"-{np.random.randint(5, 15)}"
        )
    
    with col2:
        st.metric(
            "Medium Risk Customers", 
            churn_data['medium_risk'],
            f"+{np.random.randint(1, 8)}"
        )
    
    with col3:
        st.metric(
            "Low Risk Customers",
            churn_data['low_risk'],
            f"+{np.random.randint(10, 25)}"
        )
    
    # Risk distribution
    risk_dist = pd.DataFrame({
        'Risk Level': ['High', 'Medium', 'Low'],
        'Customers': [churn_data['high_risk'], churn_data['medium_risk'], churn_data['low_risk']],
        'Percentage': [25, 35, 40]
    })
    
    fig_risk = px.bar(
        risk_dist,
        x='Risk Level',
        y='Customers',
        color='Risk Level',
        color_discrete_map={'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'},
        title='Customer Churn Risk Distribution'
    )
    st.plotly_chart(fig_risk, use_container_width=True)

def show_model_overview():
    """Display ML model overview and performance"""
    st.subheader("🤖 ML Model Portfolio")
    
    # Model cards
    models = [
        {
            'name': 'Sales Forecasting Model',
            'type': 'Time Series (ARIMA + ML)',
            'accuracy': 94.2,
            'last_trained': '2 days ago',
            'status': 'Active',
            'predictions_today': 1247
        },
        {
            'name': 'Customer Segmentation Model',
            'type': 'K-Means Clustering',
            'accuracy': 89.7,
            'last_trained': '1 week ago',
            'status': 'Active',
            'predictions_today': 856
        },
        {
            'name': 'Churn Prediction Model',
            'type': 'Random Forest',
            'accuracy': 91.5,
            'last_trained': '3 days ago',
            'status': 'Active',
            'predictions_today': 2134
        },
        {
            'name': 'Price Optimization Model',
            'type': 'XGBoost',
            'accuracy': 87.3,
            'last_trained': '5 days ago',
            'status': 'Scheduled',
            'predictions_today': 0
        }
    ]
    
    col1, col2 = st.columns(2)
    
    for i, model in enumerate(models):
        with col1 if i % 2 == 0 else col2:
            with st.container():
                st.markdown(f"#### {model['name']}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Type:** {model['type']}")
                    st.write(f"**Accuracy:** {model['accuracy']}%")
                with col_b:
                    st.write(f"**Status:** {model['status']}")
                    st.write(f"**Last Trained:** {model['last_trained']}")
                
                st.write(f"**Predictions Today:** {model['predictions_today']:,}")
                
                # Status indicator
                if model['status'] == 'Active':
                    st.success("🟢 Model is running")
                else:
                    st.warning("🟡 Model scheduled for training")
                
                st.markdown("---")
    
    # Training pipeline status
    st.subheader("🔄 Training Pipeline Status")
    
    pipeline_data = {
        'Pipeline': ['Data Ingestion', 'Data Preprocessing', 'Feature Engineering', 'Model Training', 'Model Validation', 'Deployment'],
        'Status': ['Completed', 'In Progress', 'Queued', 'Queued', 'Queued', 'Queued'],
        'Duration': ['2 min', '5 min', '-', '-', '-', '-'],
        'Progress': [100, 65, 0, 0, 0, 0]
    }
    
    pipeline_df = pd.DataFrame(pipeline_data)
    
    for _, row in pipeline_df.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
        with col1:
            st.write(row['Pipeline'])
        
        with col2:
            if row['Status'] == 'Completed':
                st.success('✅ Done')
            elif row['Status'] == 'In Progress':
                st.info('🔄 Running')
            else:
                st.warning('⏳ Queued')
        
        with col3:
            st.write(row['Duration'])
        
        with col4:
            if row['Progress'] > 0:
                st.progress(row['Progress'] / 100)

if __name__ == "__main__":
    main()

  
