import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_generator import DataGenerator
from utils.ml_models import MLModels
from utils.metrics import MetricsCalculator

st.set_page_config(page_title="ML Workflows", page_icon="🔄", layout="wide")

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'metrics_calc' not in st.session_state:
    st.session_state.metrics_calc = MetricsCalculator()

st.title("🔄 ML Workflows & Pipeline Execution")
st.markdown("### Automated ML Pipeline Management")
st.markdown("---")

# Pipeline selection
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Available ML Pipelines")
    
    pipeline_type = st.selectbox(
        "Select Pipeline Type",
        ["Sales Forecasting", "Customer Segmentation", "Churn Prediction", "Price Optimization"]
    )

with col2:
    st.subheader("⚙️ Pipeline Controls")
    if st.button("▶️ Run Pipeline", type="primary", use_container_width=True):
        st.session_state.pipeline_running = True
    if st.button("⏹️ Stop Pipeline", use_container_width=True):
        st.session_state.pipeline_running = False

st.markdown("---")

# Pipeline Configuration
st.subheader("🛠️ Pipeline Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    if pipeline_type == "Sales Forecasting":
        forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        model_type = st.selectbox("Model Type", ["ARIMA", "Prophet", "LSTM", "XGBoost"])
        train_split = st.slider("Train/Test Split", 0.6, 0.9, 0.8)

with col2:
    if pipeline_type == "Customer Segmentation":
        n_clusters = st.slider("Number of Segments", 3, 10, 5)
        features = st.multiselect(
            "Features to Use",
            ["CLV", "Purchase Frequency", "Recency", "AOV", "Geography"],
            default=["CLV", "Purchase Frequency", "Recency"]
        )
        algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])

with col3:
    if pipeline_type == "Churn Prediction":
        prediction_window = st.slider("Prediction Window (days)", 30, 180, 90)
        threshold = st.slider("Churn Threshold", 0.3, 0.8, 0.5)
        algorithm = st.selectbox("Algorithm", ["Random Forest", "XGBoost", "Logistic Regression"])

st.markdown("---")

# Pipeline Execution Status
st.subheader("📊 Pipeline Execution Status")

if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False

if st.session_state.pipeline_running:
    # Show pipeline stages
    stages = [
        {"name": "Data Ingestion", "status": "✅ Completed", "duration": "2m 15s", "progress": 100},
        {"name": "Data Validation", "status": "✅ Completed", "duration": "1m 30s", "progress": 100},
        {"name": "Feature Engineering", "status": "🔄 In Progress", "duration": "3m 45s", "progress": 65},
        {"name": "Model Training", "status": "⏳ Queued", "duration": "-", "progress": 0},
        {"name": "Model Evaluation", "status": "⏳ Queued", "duration": "-", "progress": 0},
        {"name": "Model Deployment", "status": "⏳ Queued", "duration": "-", "progress": 0}
    ]
    
    for stage in stages:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
        with col1:
            st.write(f"**{stage['name']}**")
        with col2:
            st.write(stage['status'])
        with col3:
            st.write(stage['duration'])
        with col4:
            if stage['progress'] > 0:
                st.progress(stage['progress'] / 100)
    
    st.info("💡 Pipeline is currently running. Estimated completion: 8 minutes")

# Pipeline Results
if pipeline_type == "Sales Forecasting" and st.session_state.pipeline_running:
    st.markdown("---")
    st.subheader("📈 Pipeline Results")
    
    results = st.session_state.ml_models.run_sales_pipeline(
        horizon=forecast_horizon if 'forecast_horizon' in locals() else 30,
        model_type=model_type if 'model_type' in locals() else "XGBoost",
        split=train_split if 'train_split' in locals() else 0.8
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{results['mae']:.2f}", f"{results['mae_delta']:.1f}%")
    with col2:
        st.metric("RMSE", f"{results['rmse']:.2f}", f"{results['rmse_delta']:.1f}%")
    with col3:
        st.metric("MAPE", f"{results['mape']:.2f}%", f"{results['mape_delta']:.1f}%")
    with col4:
        st.metric("R² Score", f"{results['r2']:.3f}", f"{results['r2_delta']:.3f}")
    
    # Actual vs Predicted chart
    fig = go.Figure()
    
    sample_indices = np.random.choice(len(results['actual_values']), 100, replace=False)
    sample_indices.sort()
    
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=[results['actual_values'][i] for i in sample_indices],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=[results['predicted_values'][i] for i in sample_indices],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Sample Index',
        yaxis_title='Value',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif pipeline_type == "Customer Segmentation" and st.session_state.pipeline_running:
    st.markdown("---")
    st.subheader("🎯 Segmentation Results")
    
    results = st.session_state.ml_models.run_segmentation_pipeline(
        n_clusters=n_clusters if 'n_clusters' in locals() else 5,
        features=features if 'features' in locals() else ["CLV", "Purchase Frequency"],
        algorithm=algorithm if 'algorithm' in locals() else "K-Means"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
    with col2:
        st.metric("Inertia", f"{results['inertia']:.2f}")
    with col3:
        st.metric("Calinski-Harabasz", f"{results['calinski_score']:.2f}")
    
    # Segment characteristics
    st.subheader("📊 Segment Characteristics")
    
    segment_df = pd.DataFrame(results['segment_characteristics'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=segment_df['segment'],
        y=segment_df['size'],
        name='Segment Size',
        marker_color='#45B7D1'
    ))
    
    fig.update_layout(
        title='Customer Segment Distribution',
        xaxis_title='Segment',
        yaxis_title='Number of Customers',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(segment_df, use_container_width=True)

elif pipeline_type == "Churn Prediction" and st.session_state.pipeline_running:
    st.markdown("---")
    st.subheader("🚨 Churn Prediction Results")
    
    results = st.session_state.ml_models.run_churn_pipeline(
        algorithm=algorithm if 'algorithm' in locals() else "Random Forest",
        window=prediction_window if 'prediction_window' in locals() else 90,
        threshold=threshold if 'threshold' in locals() else 0.5
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{results['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{results['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{results['f1_score']:.2%}")
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    
    feature_df = pd.DataFrame({
        'Feature': list(results['feature_importance'].keys()),
        'Importance': list(results['feature_importance'].values())
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        feature_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top Risk Factors',
        color='Importance',
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

elif pipeline_type == "Price Optimization" and st.session_state.pipeline_running:
    st.markdown("---")
    st.subheader("💰 Price Optimization Results")
    
    objective = st.radio("Optimization Objective", ["Maximize Revenue", "Maximize Profit", "Maximize Market Share"])
    constraints = st.multiselect("Constraints", ["Min Price", "Max Price", "Competitor Pricing", "Cost Plus"])
    category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"])
    
    results = st.session_state.ml_models.run_price_optimization(
        objective=objective,
        constraints=constraints,
        category=category
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Optimal Price", f"${results['optimal_price']:.2f}", f"{results['price_change']:.1%}")
    with col2:
        st.metric("Expected Revenue", f"${results['expected_revenue']:,.0f}", f"{results['revenue_change']:.1%}")
    with col3:
        st.metric("Profit Margin", f"{results['profit_margin']:.1%}", f"{results['margin_change']:.1%}")

# Recent Pipeline History
st.markdown("---")
st.subheader("📜 Recent Pipeline Executions")

history_data = {
    'Timestamp': [
        (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') 
        for i in [1, 3, 6, 12, 24]
    ],
    'Pipeline': ['Sales Forecasting', 'Churn Prediction', 'Customer Segmentation', 'Price Optimization', 'Sales Forecasting'],
    'Status': ['✅ Success', '✅ Success', '⚠️ Warning', '✅ Success', '✅ Success'],
    'Duration': ['12m 34s', '8m 45s', '15m 12s', '6m 23s', '11m 56s'],
    'Accuracy': ['94.2%', '91.5%', '89.1%', '92.8%', '93.7%']
}

history_df = pd.DataFrame(history_data)
st.dataframe(history_df, use_container_width=True)
