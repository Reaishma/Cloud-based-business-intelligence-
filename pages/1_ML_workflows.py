import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
from utils.data_generator import DataGenerator
from utils.ml_models import MLModels

st.set_page_config(page_title="ML Workflows", page_icon="🔬", layout="wide")

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()

def main():
    st.title("🔬 ML Workflows & Pipeline Management")
    st.markdown("### Simulated Amazon SageMaker Workflows")
    st.markdown("---")
    
    # Workflow selection
    workflow_type = st.selectbox(
        "Select ML Workflow",
        ["Sales Forecasting Pipeline", "Customer Segmentation Pipeline", "Churn Prediction Pipeline", "Price Optimization Pipeline"]
    )
    
    if workflow_type == "Sales Forecasting Pipeline":
        sales_forecasting_workflow()
    elif workflow_type == "Customer Segmentation Pipeline":
        customer_segmentation_workflow()
    elif workflow_type == "Churn Prediction Pipeline":
        churn_prediction_workflow()
    else:
        price_optimization_workflow()

def sales_forecasting_workflow():
    """Sales forecasting ML workflow simulation"""
    st.subheader("📈 Sales Forecasting Pipeline")
    
    # Pipeline configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Pipeline Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            ["Sales Database", "Data Warehouse", "S3 Bucket", "API Feed"]
        )
        
        # Model parameters
        forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        model_type = st.selectbox(
            "Model Type",
            ["ARIMA + ML Hybrid", "Prophet", "LSTM", "XGBoost Time Series"]
        )
        
        # Training parameters
        train_split = st.slider("Training Split", 0.6, 0.9, 0.8)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        # Execute pipeline button
        if st.button("🚀 Execute Pipeline", type="primary"):
            execute_sales_pipeline(forecast_horizon, model_type, train_split)
    
    with col2:
        st.markdown("#### Pipeline Execution Log")
        
        # Real-time pipeline status
        if st.button("View Live Execution"):
            show_pipeline_execution()
        
        # Show workflow diagram
        show_workflow_diagram()

def show_pipeline_execution():
    """Simulate real-time pipeline execution"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()
    
    steps = [
        "Initializing SageMaker training job...",
        "Loading data from source...",
        "Preprocessing data...",
        "Feature engineering...",
        "Splitting data into train/validation...",
        "Training model...",
        "Validating model performance...",
        "Generating predictions...",
        "Saving model artifacts...",
        "Deploying model endpoint...",
        "Pipeline completed successfully!"
    ]
    
    logs = []
    
    for i, step in enumerate(steps):
        progress = (i + 1) / len(steps)
        progress_bar.progress(progress)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {step}"
        logs.append(log_entry)
        
        status_text.text(f"Current step: {step}")
        log_container.text_area("Execution Log", "\n".join(logs), height=200)
        
        time.sleep(0.5)  # Simulate processing time
    
    st.success("✅ Pipeline execution completed successfully!")

def execute_sales_pipeline(horizon, model_type, split):
    """Execute the sales forecasting pipeline"""
    st.markdown("#### Execution Results")
    
    # Generate synthetic results
    results = st.session_state.ml_models.run_sales_pipeline(horizon, model_type, split)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{results['mae']:.2f}", f"{results['mae_delta']:.1f}%")
    
    with col2:
        st.metric("RMSE", f"{results['rmse']:.2f}", f"{results['rmse_delta']:.1f}%")
    
    with col3:
        st.metric("MAPE", f"{results['mape']:.1f}%", f"{results['mape_delta']:.1f}%")
    
    with col4:
        st.metric("R²", f"{results['r2']:.3f}", f"{results['r2_delta']:.2f}")
    
    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Actual vs Predicted', 'Residuals', 'Feature Importance', 'Error Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Actual vs Predicted
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    actual = results['actual_values']
    predicted = results['predicted_values']
    
    fig.add_trace(
        go.Scatter(x=dates, y=actual, mode='lines', name='Actual', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted', line=dict(color='red')),
        row=1, col=1
    )
    
    # Residuals
    residuals = np.array(actual) - np.array(predicted)
    fig.add_trace(
        go.Scatter(x=dates, y=residuals, mode='markers', name='Residuals'),
        row=1, col=2
    )
    
    # Feature importance
    features = ['Seasonality', 'Trend', 'Marketing Spend', 'Price', 'Competition']
    importance = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    fig.add_trace(
        go.Bar(x=features, y=importance, name='Importance'),
        row=2, col=1
    )
    
    # Error distribution
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=20, name='Error Distribution'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def customer_segmentation_workflow():
    """Customer segmentation ML workflow"""
    st.subheader("👥 Customer Segmentation Pipeline")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Segmentation Parameters")
        
        n_clusters = st.slider("Number of Segments", 2, 10, 5)
        features = st.multiselect(
            "Select Features",
            ["Purchase Frequency", "Average Order Value", "Customer Lifetime Value", 
             "Recency", "Geographic Location", "Product Preferences"],
            default=["Purchase Frequency", "Average Order Value", "Customer Lifetime Value"]
        )
        
        algorithm = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture"]
        )
        
        if st.button("🎯 Run Segmentation", type="primary"):
            run_segmentation_pipeline(n_clusters, features, algorithm)
    
    with col2:
        st.markdown("#### Segmentation Results")
        
        # Generate segmentation visualization
        segment_data = st.session_state.ml_models.generate_customer_segments(n_clusters)
        
        fig = px.scatter_3d(
            segment_data,
            x='feature_1',
            y='feature_2', 
            z='feature_3',
            color='segment',
            title='Customer Segments (3D Visualization)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def run_segmentation_pipeline(n_clusters, features, algorithm):
    """Run customer segmentation pipeline"""
    
    # Show progress
    with st.spinner("Running segmentation analysis..."):
        time.sleep(2)  # Simulate processing
    
    # Results
    results = st.session_state.ml_models.run_segmentation_pipeline(n_clusters, features, algorithm)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
    
    with col2:
        st.metric("Inertia", f"{results['inertia']:.0f}")
    
    with col3:
        st.metric("Calinski-Harabasz", f"{results['calinski_score']:.1f}")
    
    # Segment characteristics
    st.markdown("#### Segment Characteristics")
    
    segment_chars = pd.DataFrame(results['segment_characteristics'])
    
    fig = px.parallel_coordinates(
        segment_chars,
        color='segment',
        title='Segment Characteristics Parallel Plot'
    )
    st.plotly_chart(fig, use_container_width=True)

def churn_prediction_workflow():
    """Churn prediction ML workflow"""
    st.subheader("🚨 Churn Prediction Pipeline")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Model Configuration")
        
        model_algorithm = st.selectbox(
            "Algorithm",
            ["Random Forest", "XGBoost", "Logistic Regression", "Neural Network"]
        )
        
        prediction_window = st.selectbox(
            "Prediction Window",
            ["7 days", "30 days", "90 days", "180 days"]
        )
        
        threshold = st.slider("Churn Probability Threshold", 0.1, 0.9, 0.5)
        
        if st.button("🔍 Run Churn Analysis", type="primary"):
            run_churn_pipeline(model_algorithm, prediction_window, threshold)
    
    with col2:
        st.markdown("#### Churn Risk Analysis")
        
        # Generate churn analysis
        churn_results = st.session_state.ml_models.analyze_churn_risk()
        
        # Risk distribution
        fig = go.Figure(data=[
            go.Bar(name='Low Risk', x=['Customers'], y=[churn_results['low_risk']], marker_color='green'),
            go.Bar(name='Medium Risk', x=['Customers'], y=[churn_results['medium_risk']], marker_color='orange'),
            go.Bar(name='High Risk', x=['Customers'], y=[churn_results['high_risk']], marker_color='red')
        ])
        
        fig.update_layout(
            title='Churn Risk Distribution',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def run_churn_pipeline(algorithm, window, threshold):
    """Execute churn prediction pipeline"""
    
    with st.spinner("Training churn prediction model..."):
        time.sleep(3)  # Simulate training
    
    results = st.session_state.ml_models.run_churn_pipeline(algorithm, window, threshold)
    
    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.1%}")
    
    with col2:
        st.metric("Precision", f"{results['precision']:.1%}")
    
    with col3:
        st.metric("Recall", f"{results['recall']:.1%}")
    
    with col4:
        st.metric("F1 Score", f"{results['f1_score']:.1%}")
    
    # Feature importance
    st.markdown("#### Feature Importance")
    
    features = results['feature_importance']
    fig = px.bar(
        x=list(features.values()),
        y=list(features.keys()),
        orientation='h',
        title='Top Features for Churn Prediction'
    )
    st.plotly_chart(fig, use_container_width=True)

def price_optimization_workflow():
    """Price optimization ML workflow"""
    st.subheader("💰 Price Optimization Pipeline")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Optimization Parameters")
        
        objective = st.selectbox(
            "Optimization Objective",
            ["Maximize Revenue", "Maximize Profit", "Maximize Market Share", "Minimize Churn"]
        )
        
        constraints = st.multiselect(
            "Constraints",
            ["Competitor Pricing", "Cost Margins", "Demand Elasticity", "Inventory Levels"],
            default=["Cost Margins", "Demand Elasticity"]
        )
        
        product_category = st.selectbox(
            "Product Category",
            ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
        )
        
        if st.button("🎯 Optimize Pricing", type="primary"):
            run_price_optimization(objective, constraints, product_category)
    
    with col2:
        st.markdown("#### Price Elasticity Analysis")
        
        # Generate price elasticity curve
        prices = np.linspace(10, 100, 50)
        demand = 1000 * np.exp(-0.02 * prices) + np.random.normal(0, 10, 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices,
            y=demand,
            mode='lines+markers',
            name='Demand Curve',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title='Price-Demand Elasticity',
            xaxis_title='Price ($)',
            yaxis_title='Demand (units)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def run_price_optimization(objective, constraints, category):
    """Execute price optimization pipeline"""
    
    with st.spinner("Running price optimization algorithm..."):
        time.sleep(2)
    
    results = st.session_state.ml_models.run_price_optimization(objective, constraints, category)
    
    # Optimization results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Optimal Price", f"${results['optimal_price']:.2f}", f"{results['price_change']:.1%}")
    
    with col2:
        st.metric("Expected Revenue", f"${results['expected_revenue']:,.0f}", f"{results['revenue_change']:.1%}")
    
    with col3:
        st.metric("Profit Margin", f"{results['profit_margin']:.1%}", f"{results['margin_change']:.1%}")
    
    # Sensitivity analysis
    st.markdown("#### Price Sensitivity Analysis")
    
    price_range = np.linspace(results['optimal_price'] * 0.8, results['optimal_price'] * 1.2, 20)
    revenue_impact = [results['expected_revenue'] * (1 + np.random.uniform(-0.15, 0.15)) for _ in price_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_range,
        y=revenue_impact,
        mode='lines+markers',
        name='Revenue Impact',
        line=dict(color='green', width=3)
    ))
    
    # Mark optimal point
    fig.add_trace(go.Scatter(
        x=[results['optimal_price']],
        y=[results['expected_revenue']],
        mode='markers',
        name='Optimal Point',
        marker=dict(color='red', size=12, symbol='star')
    ))
    
    fig.update_layout(
        title='Revenue Sensitivity to Price Changes',
        xaxis_title='Price ($)',
        yaxis_title='Expected Revenue ($)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_workflow_diagram():
    """Display workflow architecture diagram"""
    st.markdown("#### Workflow Architecture")
    
    # Create a simple workflow diagram using plotly
    fig = go.Figure()
    
    # Define workflow steps
    steps = [
        {"name": "Data Source", "x": 1, "y": 3, "color": "#FF6B6B"},
        {"name": "Data Processing", "x": 2, "y": 3, "color": "#4ECDC4"},
        {"name": "Feature Engineering", "x": 3, "y": 3, "color": "#45B7D1"},
        {"name": "Model Training", "x": 4, "y": 3, "color": "#96CEB4"},
        {"name": "Validation", "x": 5, "y": 3, "color": "#FFEAA7"},
        {"name": "Deployment", "x": 6, "y": 3, "color": "#DDA0DD"}
    ]
    
    # Add workflow steps
    for step in steps:
        fig.add_trace(go.Scatter(
            x=[step["x"]],
            y=[step["y"]],
            mode='markers+text',
            marker=dict(size=50, color=step["color"]),
            text=step["name"],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            showlegend=False
        ))
    
    # Add arrows between steps
    for i in range(len(steps) - 1):
        fig.add_trace(go.Scatter(
            x=[steps[i]["x"] + 0.3, steps[i+1]["x"] - 0.3],
            y=[steps[i]["y"], steps[i+1]["y"]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title="ML Pipeline Workflow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=200,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
      
