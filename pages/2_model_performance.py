import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from utils.ml_models import MLModels
from utils.metrics import MetricsCalculator

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")

# Initialize session state
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'metrics_calc' not in st.session_state:
    st.session_state.metrics_calc = MetricsCalculator()

def main():
    st.title("📊 Model Performance Analytics")
    st.markdown("### Comprehensive ML Model Evaluation & Monitoring")
    st.markdown("---")
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model for Analysis",
        ["Sales Forecasting Model", "Customer Segmentation Model", "Churn Prediction Model", "Price Optimization Model"]
    )
    
    # Performance overview
    show_performance_overview(selected_model)
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Metrics", 
        "🎯 Model Validation", 
        "📊 Feature Analysis", 
        "⚡ Real-time Monitoring", 
        "🔄 Model Comparison"
    ])
    
    with tab1:
        show_performance_metrics(selected_model)
    
    with tab2:
        show_model_validation(selected_model)
    
    with tab3:
        show_feature_analysis(selected_model)
    
    with tab4:
        show_realtime_monitoring(selected_model)
    
    with tab5:
        show_model_comparison()

def show_performance_overview(model_name):
    """Display high-level performance overview"""
    st.subheader(f"🎯 {model_name} - Performance Overview")
    
    # Get model performance data
    perf_data = st.session_state.metrics_calc.get_model_performance(model_name)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Accuracy",
            f"{perf_data['accuracy']:.1%}",
            f"{perf_data['accuracy_delta']:.1%}"
        )
    
    with col2:
        st.metric(
            "Model Precision",
            f"{perf_data['precision']:.1%}",
            f"{perf_data['precision_delta']:.1%}"
        )
    
    with col3:
        st.metric(
            "Model Recall",
            f"{perf_data['recall']:.1%}",
            f"{perf_data['recall_delta']:.1%}"
        )
    
    with col4:
        st.metric(
            "F1 Score",
            f"{perf_data['f1_score']:.1%}",
            f"{perf_data['f1_delta']:.1%}"
        )
    
    # Performance trend
    st.markdown("#### Performance Trend (Last 30 Days)")
    
    dates = pd.date_range(start='2024-06-01', end='2024-06-30', freq='D')
    accuracy_trend = np.random.normal(perf_data['accuracy'], 0.02, len(dates))
    accuracy_trend = np.clip(accuracy_trend, 0.8, 0.98)  # Keep realistic bounds
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=accuracy_trend,
        mode='lines+markers',
        name='Daily Accuracy',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=6)
    ))
    
    # Add trend line
    z = np.polyfit(range(len(dates)), accuracy_trend, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=dates,
        y=p(range(len(dates))),
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Model Accuracy Over Time',
        xaxis_title='Date',
        yaxis_title='Accuracy',
        height=400,
        yaxis=dict(tickformat='.1%')
    )
    st.plotly_chart(fig, use_container_width=True)

def show_performance_metrics(model_name):
    """Display detailed performance metrics"""
    st.subheader("📈 Detailed Performance Metrics")
    
    # Generate comprehensive metrics
    metrics = st.session_state.metrics_calc.get_detailed_metrics(model_name)
    
    # Classification metrics (for classification models)
    if model_name in ["Churn Prediction Model", "Customer Segmentation Model"]:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Confusion Matrix")
            
            # Generate confusion matrix
            cm = metrics['confusion_matrix']
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                title='Confusion Matrix'
            )
            fig.update_layout(
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ROC Curve")
            
            # Generate ROC curve data
            fpr = metrics['roc_curve']['fpr']
            tpr = metrics['roc_curve']['tpr']
            auc = metrics['roc_curve']['auc']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc:.3f})',
                line=dict(color='blue', width=3)
            ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Precision-Recall curve
        st.markdown("#### Precision-Recall Curve")
        
        precision = metrics['pr_curve']['precision']
        recall = metrics['pr_curve']['recall']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Regression metrics (for regression models)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Prediction vs Actual")
            
            actual = metrics['actual_values']
            predicted = metrics['predicted_values']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual,
                y=predicted,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            
            # Add perfect prediction line
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Predicted vs Actual Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Residuals Analysis")
            
            residuals = np.array(actual) - np.array(predicted)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predicted,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='orange', size=8, opacity=0.6)
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title='Residuals vs Predicted Values',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        st.markdown("#### Error Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Error Distribution',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Distribution of Prediction Errors',
            xaxis_title='Error',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_validation(model_name):
    """Display model validation results"""
    st.subheader("🎯 Model Validation & Cross-Validation")
    
    # Cross-validation results
    cv_results = st.session_state.metrics_calc.get_cv_results(model_name)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cross-Validation Scores")
        
        cv_scores = cv_results['cv_scores']
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=cv_scores,
            name='CV Scores',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'Cross-Validation Scores (Mean: {cv_mean:.3f} ± {cv_std:.3f})',
            yaxis_title='Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Validation metrics table
        st.markdown("#### Validation Metrics")
        validation_df = pd.DataFrame({
            'Metric': ['Mean CV Score', 'Std CV Score', 'Min CV Score', 'Max CV Score'],
            'Value': [cv_mean, cv_std, min(cv_scores), max(cv_scores)]
        })
        st.dataframe(validation_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Learning Curves")
        
        # Generate learning curve data
        train_sizes = cv_results['train_sizes']
        train_scores = cv_results['train_scores']
        val_scores = cv_results['val_scores']
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue', width=3),
            error_y=dict(
                type='data',
                array=np.std(train_scores, axis=1),
                visible=True
            )
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red', width=3),
            error_y=dict(
                type='data',
                array=np.std(val_scores, axis=1),
                visible=True
            )
        ))
        
        fig.update_layout(
            title='Learning Curves',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Hyperparameter tuning results
    st.markdown("#### Hyperparameter Tuning Results")
    
    tuning_results = cv_results['hyperparameter_tuning']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Best parameters
        st.markdown("**Best Parameters:**")
        for param, value in tuning_results['best_params'].items():
            st.write(f"• {param}: {value}")
    
    with col2:
        # Parameter importance
        param_importance = tuning_results['param_importance']
        
        fig = px.bar(
            x=list(param_importance.values()),
            y=list(param_importance.keys()),
            orientation='h',
            title='Hyperparameter Importance'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(model_name):
    """Display feature importance and analysis"""
    st.subheader("📊 Feature Analysis & Importance")
    
    # Feature importance
    feature_data = st.session_state.metrics_calc.get_feature_analysis(model_name)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature Importance")
        
        features = list(feature_data['importance'].keys())
        importance = list(feature_data['importance'].values())
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Feature Importance Ranking',
            color=importance,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Feature Correlation Matrix")
        
        # Generate correlation matrix
        correlation_matrix = feature_data['correlation_matrix']
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title='Feature Correlation Matrix'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distribution analysis
    st.markdown("#### Feature Distribution Analysis")
    
    selected_features = st.multiselect(
        "Select features to analyze",
        features,
        default=features[:3]
    )
    
    if selected_features:
        feature_distributions = feature_data['distributions']
        
        fig = make_subplots(
            rows=1, cols=len(selected_features),
            subplot_titles=selected_features
        )
        
        for i, feature in enumerate(selected_features):
            fig.add_trace(
                go.Histogram(
                    x=feature_distributions[feature],
                    name=feature,
                    nbinsx=20
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Feature Distributions',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature interaction analysis
    st.markdown("#### Feature Interactions")
    
    if len(selected_features) >= 2:
        feature1 = st.selectbox("Select first feature", selected_features)
        feature2 = st.selectbox("Select second feature", [f for f in selected_features if f != feature1])

  if feature1 and feature2:
            # Generate interaction plot
            x_data = feature_distributions[feature1]
            y_data = feature_distributions[feature2]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name='Feature Interaction',
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color=np.random.randn(len(x_data)),
                    colorscale='viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title=f'{feature1} vs {feature2} Interaction',
                xaxis_title=feature1,
                yaxis_title=feature2,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def show_realtime_monitoring(model_name):
    """Display real-time model monitoring"""
    st.subheader("⚡ Real-time Model Monitoring")
    
    # Model health status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predictions/Hour", "1,247", "+12%")
    
    with col2:
        st.metric("Response Time", "45ms", "-2ms")
    
    with col3:
        st.metric("Error Rate", "0.02%", "-0.01%")
    
    with col4:
        st.metric("Model Uptime", "99.8%", "+0.1%")
    
    # Real-time performance monitoring
    st.markdown("#### Real-Time Performance Monitoring")
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("Auto-refresh data", value=False)
    
    if auto_refresh:
        st.rerun()
    
    # Generate real-time data
    monitoring_data = st.session_state.metrics_calc.get_realtime_monitoring(model_name)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time monitoring
        timestamps = pd.date_range(start='now', periods=100, freq='1min')
        response_times = monitoring_data['response_times']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=response_times,
            mode='lines',
            name='Response Time',
            line=dict(color='blue', width=2)
        ))
        
        # Add threshold line
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                     annotation_text="SLA Threshold")
        
        fig.update_layout(
            title='Model Response Time (Last 100 Minutes)',
            xaxis_title='Time',
            yaxis_title='Response Time (ms)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction volume
        prediction_volume = monitoring_data['prediction_volume']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prediction_volume,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Prediction Volume (Last 100 Minutes)',
            xaxis_title='Time',
            yaxis_title='Predictions per Minute',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data drift detection
    st.markdown("#### Data Drift Detection")
    
    drift_data = monitoring_data['data_drift']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature drift scores
        drift_scores = drift_data['drift_scores']
        
        fig = px.bar(
            x=list(drift_scores.keys()),
            y=list(drift_scores.values()),
            title='Feature Drift Scores',
            color=list(drift_scores.values()),
            color_continuous_scale='reds'
        )
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Drift Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance over time
        performance_history = drift_data['performance_history']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(performance_history))),
            y=performance_history,
            mode='lines+markers',
            name='Model Performance',
            line=dict(color='purple', width=3)
        ))
        
        fig.update_layout(
            title='Model Performance Over Time',
            xaxis_title='Time Period',
            yaxis_title='Performance Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Alerts and notifications
    st.markdown("#### Recent Alerts")
    
    alerts = monitoring_data['alerts']
    
    for alert in alerts:
        if alert['severity'] == 'high':
            st.error(f"🚨 {alert['message']} - {alert['timestamp']}")
        elif alert['severity'] == 'medium':
            st.warning(f"⚠️ {alert['message']} - {alert['timestamp']}")
        else:
            st.info(f"ℹ️ {alert['message']} - {alert['timestamp']}")

def show_model_comparison():
    """Display model comparison analysis"""
    st.subheader("🔄 Model Comparison Analysis")
    
    # Model selection for comparison
    models = ["Sales Forecasting Model", "Customer Segmentation Model", "Churn Prediction Model", "Price Optimization Model"]
    selected_models = st.multiselect("Select models to compare", models, default=models[:3])
    
    if len(selected_models) >= 2:
        
        # Performance comparison
        st.markdown("#### Performance Comparison")
        
        comparison_data = st.session_state.metrics_calc.compare_models(selected_models)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data['performance_metrics'])
        
        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for model in selected_models:
            fig.add_trace(go.Scatterpolar(
                r=[comparison_data['performance_metrics'][model][metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='Model Performance Comparison (Radar Chart)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource utilization comparison
        st.markdown("#### Resource Utilization Comparison")
        
        resource_data = comparison_data['resource_utilization']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Training time comparison
            training_times = [resource_data[model]['training_time'] for model in selected_models]
            
            fig = px.bar(
                x=selected_models,
                y=training_times,
                title='Training Time Comparison',
                color=training_times,
                color_continuous_scale='viridis'
            )
            fig.update_layout(yaxis_title='Training Time (minutes)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Memory usage comparison
            memory_usage = [resource_data[model]['memory_usage'] for model in selected_models]
            
            fig = px.bar(
                x=selected_models,
                y=memory_usage,
                title='Memory Usage Comparison',
                color=memory_usage,
                color_continuous_scale='reds'
            )
            fig.update_layout(yaxis_title='Memory Usage (MB)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model complexity comparison
        st.markdown("#### Model Complexity Analysis")
        
        complexity_metrics = ['Number of Features', 'Model Parameters', 'Training Data Size']
        
        fig = make_subplots(
            rows=1, cols=len(complexity_metrics),
            subplot_titles=complexity_metrics
        )
        
        for i, metric in enumerate(complexity_metrics):
            values = [comparison_data['complexity'][model][metric] for model in selected_models]
            
            fig.add_trace(
                go.Bar(x=selected_models, y=values, name=metric),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Model Complexity Comparison',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
  
