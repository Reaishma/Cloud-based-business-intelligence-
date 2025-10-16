import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.data_generator import DataGenerator
from utils.ml_models import MLModels
from utils.metrics import MetricsCalculator

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'metrics_calc' not in st.session_state:
    st.session_state.metrics_calc = MetricsCalculator()

st.title("📊 Model Performance & Monitoring")
st.markdown("### Real-time Model Analytics and Metrics")
st.markdown("---")

# Model selection
col1, col2 = st.columns([2, 1])

with col1:
    selected_model = st.selectbox(
        "Select Model to Analyze",
        [
            "Sales Forecasting Model",
            "Customer Segmentation Model",
            "Churn Prediction Model",
            "Price Optimization Model"
        ]
    )

with col2:
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )

st.markdown("---")

# Key Performance Metrics
st.subheader("🎯 Key Performance Indicators")

metrics = st.session_state.metrics_calc.get_model_performance(selected_model)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}",
        f"{metrics['accuracy_delta']:.1%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{metrics['precision']:.2%}",
        f"{metrics['precision_delta']:.1%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{metrics['recall']:.2%}",
        f"{metrics['recall_delta']:.1%}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{metrics['f1_score']:.2%}",
        f"{metrics['f1_delta']:.1%}"
    )

st.markdown("---")

# Performance Visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Model Performance Over Time")
    
    # Generate performance history
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'Accuracy': np.random.uniform(0.88, 0.95, 30),
        'Precision': np.random.uniform(0.85, 0.93, 30),
        'Recall': np.random.uniform(0.82, 0.91, 30)
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_data['Date'],
        y=performance_data['Accuracy'],
        mode='lines',
        name='Accuracy',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=performance_data['Date'],
        y=performance_data['Precision'],
        mode='lines',
        name='Precision',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=performance_data['Date'],
        y=performance_data['Recall'],
        mode='lines',
        name='Recall',
        line=dict(color='#45B7D1', width=2)
    ))
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        yaxis_title='Score',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔄 Model Prediction Volume")
    
    # Generate prediction volume data
    hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
    volume_data = pd.DataFrame({
        'Hour': hours,
        'Predictions': np.random.poisson(150, 24) + 50 * np.sin(2 * np.pi * np.arange(24) / 24)
    })
    
    fig = px.bar(
        volume_data,
        x='Hour',
        y='Predictions',
        title='Hourly Prediction Volume',
        color='Predictions',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Detailed Metrics
st.markdown("---")
st.subheader("📋 Detailed Performance Analysis")

detailed_metrics = st.session_state.metrics_calc.get_detailed_metrics(selected_model)

tab1, tab2, tab3 = st.tabs(["🎯 Confusion Matrix", "📈 ROC Curve", "🔍 Prediction Analysis"])

with tab1:
    if 'confusion_matrix' in detailed_metrics:
        cm = detailed_metrics['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            height=400,
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            tn, fp = cm[0]
            fn, tp = cm[1]
            st.metric("True Positives", tp)
            st.metric("True Negatives", tn)
        with col2:
            st.metric("False Positives", fp)
            st.metric("False Negatives", fn)

with tab2:
    if 'roc_curve' in detailed_metrics:
        roc_data = detailed_metrics['roc_curve']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name=f'ROC Curve (AUC = {roc_data["auc"]:.3f})',
            line=dict(color='#4ECDC4', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("AUC Score", f"{roc_data['auc']:.3f}")

with tab3:
    if 'actual_values' in detailed_metrics:
        sample_size = min(100, len(detailed_metrics['actual_values']))
        indices = np.random.choice(len(detailed_metrics['actual_values']), sample_size, replace=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=indices,
            y=[detailed_metrics['actual_values'][i] for i in indices],
            mode='markers',
            name='Actual',
            marker=dict(color='#FF6B6B', size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=indices,
            y=[detailed_metrics['predicted_values'][i] for i in indices],
            mode='markers',
            name='Predicted',
            marker=dict(color='#4ECDC4', size=8)
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Sample Index',
            yaxis_title='Value',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Feature Analysis
st.markdown("---")
st.subheader("🔍 Feature Importance Analysis")

feature_analysis = st.session_state.metrics_calc.get_feature_analysis(selected_model)

col1, col2 = st.columns(2)

with col1:
    # Feature importance
    feature_df = pd.DataFrame({
        'Feature': list(feature_analysis['importance'].keys()),
        'Importance': list(feature_analysis['importance'].values())
    }).sort_values('Importance', ascending=True).tail(10)
    
    fig = px.bar(
        feature_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Feature correlation heatmap
    features = list(feature_analysis['importance'].keys())[:6]
    corr_matrix = np.array(feature_analysis['correlation_matrix'])[:6, :6]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Real-time Monitoring
st.markdown("---")
st.subheader("🚨 Real-time Monitoring & Alerts")

monitoring_data = st.session_state.metrics_calc.get_realtime_monitoring(selected_model)

col1, col2 = st.columns(2)

with col1:
    st.subheader("⏱️ Response Time Monitoring")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=monitoring_data['response_times'],
        mode='lines',
        name='Response Time',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    
    fig.update_layout(
        yaxis_title='Response Time (ms)',
        xaxis_title='Request Number',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Data Drift Detection")
    
    drift_scores = monitoring_data['data_drift']['drift_scores']
    drift_df = pd.DataFrame({
        'Feature': list(drift_scores.keys()),
        'Drift Score': list(drift_scores.values())
    })
    
    fig = px.bar(
        drift_df,
        x='Feature',
        y='Drift Score',
        title='Feature Drift Scores',
        color='Drift Score',
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Active Alerts
st.subheader("⚠️ Active Alerts")

alerts = monitoring_data['alerts']

for alert in alerts:
    if alert['severity'] == 'high':
        st.error(f"🔴 **{alert['message']}** - {alert['timestamp']}")
    elif alert['severity'] == 'medium':
        st.warning(f"🟡 **{alert['message']}** - {alert['timestamp']}")
    else:
        st.info(f"🔵 **{alert['message']}** - {alert['timestamp']}")

# Model Comparison
st.markdown("---")
st.subheader("⚖️ Model Comparison")

models_to_compare = st.multiselect(
    "Select models to compare",
    [
        "Sales Forecasting Model",
        "Customer Segmentation Model",
        "Churn Prediction Model",
        "Price Optimization Model"
    ],
    default=["Sales Forecasting Model", "Churn Prediction Model"]
)

if models_to_compare:
    comparison = st.session_state.metrics_calc.compare_models(models_to_compare)
    
    # Performance comparison
    perf_data = []
    for model, metrics in comparison['performance_metrics'].items():
        for metric, value in metrics.items():
            perf_data.append({'Model': model, 'Metric': metric, 'Value': value})
    
    perf_df = pd.DataFrame(perf_data)
    
    fig = px.bar(
        perf_df,
        x='Metric',
        y='Value',
        color='Model',
        barmode='group',
        title='Model Performance Comparison'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource utilization
    st.subheader("💻 Resource Utilization")
    
    resource_data = []
    for model, resources in comparison['resource_utilization'].items():
        resource_data.append({
            'Model': model,
            'Training Time (min)': resources['training_time'],
            'Memory (MB)': resources['memory_usage'],
            'CPU Usage (%)': resources['cpu_usage']
        })
    
    resource_df = pd.DataFrame(resource_data)
    st.dataframe(resource_df, use_container_width=True)

# Cross-Validation Results
st.markdown("---")
st.subheader("🔬 Cross-Validation Analysis")

cv_results = st.session_state.metrics_calc.get_cv_results(selected_model)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Cross-Validation Scores")
    
    cv_scores = cv_results['cv_scores']
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=cv_scores,
        name='CV Scores',
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        yaxis_title='Score',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("Mean CV Score", f"{np.mean(cv_scores):.3f}")
    st.metric("Std Dev", f"{np.std(cv_scores):.3f}")

with col2:
    st.subheader("📈 Learning Curve")
    
    fig = go.Figure()
    
    train_scores_mean = np.mean(cv_results['train_scores'], axis=1)
    val_scores_mean = np.mean(cv_results['val_scores'], axis=1)
    
    fig.add_trace(go.Scatter(
        x=cv_results['train_sizes'],
        y=train_scores_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=cv_results['train_sizes'],
        y=val_scores_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.update_layout(
        xaxis_title='Training Set Size',
        yaxis_title='Score',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Hyperparameter Tuning
st.subheader("⚙️ Hyperparameter Optimization")

tuning_results = cv_results['hyperparameter_tuning']

col1, col2 = st.columns(2)

with col1:
    st.write("**Best Parameters:**")
    for param, value in tuning_results['best_params'].items():
        st.write(f"• {param}: {value}")

with col2:
    st.write("**Parameter Importance:**")
    
    param_df = pd.DataFrame({
        'Parameter': list(tuning_results['param_importance'].keys()),
        'Importance': list(tuning_results['param_importance'].values())
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        param_df,
        x='Importance',
        y='Parameter',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Business Impact
st.markdown("---")
st.subheader("💼 Business Impact Metrics")

impact = st.session_state.metrics_calc.calculate_business_impact(selected_model)

cols = st.columns(len(impact))

for col, (metric, value) in zip(cols, impact.items()):
    with col:
        if isinstance(value, float):
            if value > 1000:
                st.metric(metric.replace('_', ' ').title(), f"${value:,.0f}")
            elif value > 1:
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
            else:
                st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")
        else:
            st.metric(metric.replace('_', ' ').title(), f"{value:,}")
