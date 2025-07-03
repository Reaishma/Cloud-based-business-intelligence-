import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.data_generator import DataGenerator
from utils.ml_models import MLModels

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide")

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()

def main():
    st.title("💡 AI-Powered Business Insights")
    st.markdown("### Strategic Intelligence & Recommendations")
    st.markdown("---")
    
    # Business insights dashboard
    show_executive_summary()
    
    # Detailed insights tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Revenue Intelligence", 
        "👥 Customer Analytics", 
        "🎯 Market Opportunities", 
        "🚀 Strategic Recommendations"
    ])
    
    with tab1:
        show_revenue_intelligence()
    
    with tab2:
        show_customer_analytics()
    
    with tab3:
        show_market_opportunities()
    
    with tab4:
        show_strategic_recommendations()

def show_executive_summary():
    """Display executive summary with key insights"""
    st.subheader("📋 Executive Summary")
    
    # Key business metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate business KPIs
    revenue_growth = np.random.uniform(8, 15)
    customer_acquisition = np.random.randint(150, 300)
    churn_rate = np.random.uniform(2, 8)
    ltv_growth = np.random.uniform(12, 25)
    
    with col1:
        st.metric(
            "Revenue Growth",
            f"{revenue_growth:.1f}%",
            f"+{np.random.uniform(1, 3):.1f}%"
        )
    
    with col2:
        st.metric(
            "New Customers",
            f"{customer_acquisition:,}",
            f"+{np.random.randint(10, 50)}"
        )
    
    with col3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%",
            f"-{np.random.uniform(0.5, 2):.1f}%"
        )
    
    with col4:
        st.metric(
            "LTV Growth",
            f"{ltv_growth:.1f}%",
            f"+{np.random.uniform(2, 5):.1f}%"
        )
    
    # AI-generated insights
    st.markdown("#### 🤖 AI-Generated Key Insights")
    
    insights = [
        {
            "title": "Revenue Acceleration Opportunity",
            "description": "ML models predict 23% revenue increase if premium product pricing is optimized for Q4",
            "impact": "High",
            "confidence": 89
        },
        {
            "title": "Customer Retention Risk",
            "description": "Churn prediction identifies 342 high-value customers at risk in the next 30 days",
            "impact": "Critical",
            "confidence": 94
        },
        {
            "title": "Market Expansion Potential",
            "description": "Geographic analysis reveals untapped markets with 45% growth potential",
            "impact": "Medium",
            "confidence": 76
        },
        {
            "title": "Product Recommendation Engine",
            "description": "Cross-selling opportunities could increase average order value by 18%",
            "impact": "High",
            "confidence": 82
        }
    ]
    
    for insight in insights:
        with st.expander(f"💡 {insight['title']} - {insight['impact']} Impact"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(insight['description'])
            
            with col2:
                # Impact indicator
                if insight['impact'] == 'Critical':
                    st.error(f"🔴 {insight['impact']}")
                elif insight['impact'] == 'High':
                    st.warning(f"🟡 {insight['impact']}")
                else:
                    st.info(f"🔵 {insight['impact']}")
                
                st.write(f"**Confidence:** {insight['confidence']}%")
                st.progress(insight['confidence'] / 100)

def show_revenue_intelligence():
    """Display revenue intelligence and forecasting"""
    st.subheader("📊 Revenue Intelligence Dashboard")
    
    # Revenue forecast
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Revenue Forecast & Trends")
        
        # Generate revenue forecast data
        forecast_data = st.session_state.ml_models.generate_revenue_forecast()
        
        fig = go.Figure()
        
        # Historical revenue
        fig.add_trace(go.Scatter(
            x=forecast_data['historical']['dates'],
            y=forecast_data['historical']['revenue'],
            mode='lines',
            name='Historical Revenue',
            line=dict(color='#2E86AB', width=3)
        ))
        
        # Forecasted revenue
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast']['dates'],
            y=forecast_data['forecast']['revenue'],
            mode='lines',
            name='Forecasted Revenue',
            line=dict(color='#A23B72', width=3, dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast']['dates'],
            y=forecast_data['forecast']['upper_bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast']['dates'],
            y=forecast_data['forecast']['lower_bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(162, 59, 114, 0.2)'
        ))
        
        fig.update_layout(
            title='6-Month Revenue Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Revenue Drivers")
        
        # Revenue breakdown
        revenue_drivers = {
            'Product Sales': 45,
            'Services': 28,
            'Subscriptions': 15,
            'Partnerships': 8,
            'Other': 4
        }
        
        fig = px.pie(
            values=list(revenue_drivers.values()),
            names=list(revenue_drivers.keys()),
            title='Revenue by Source'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        st.markdown("#### Key Metrics")
        st.metric("Monthly Recurring Revenue", "$125K", "+$12K")
        st.metric("Average Deal Size", "$2,340", "+$150")
        st.metric("Revenue per Customer", "$1,890", "+$78")
    
    # Revenue optimization opportunities
    st.markdown("#### 🎯 Revenue Optimization Opportunities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Pricing Optimization")
        
        # Price elasticity analysis
        price_points = np.linspace(50, 200, 20)
        demand_curve = 1000 * np.exp(-0.008 * price_points) + np.random.normal(0, 20, 20)
        revenue_curve = price_points * demand_curve
        
        optimal_price_idx = np.argmax(revenue_curve)
        optimal_price = price_points[optimal_price_idx]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_points,
            y=revenue_curve,
            mode='lines+markers',
            name='Revenue Curve',
            line=dict(color='green', width=3)
        ))
        
        # Mark optimal price
        fig.add_trace(go.Scatter(
            x=[optimal_price],
            y=[revenue_curve[optimal_price_idx]],
            mode='markers',
            name='Optimal Price',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig.update_layout(
            title='Price Optimization',
            xaxis_title='Price ($)',
            yaxis_title='Revenue ($)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"💡 Optimal price: ${optimal_price:.0f}")
        st.write(f"Potential revenue increase: {np.random.uniform(8, 15):.1f}%")
    
    with col2:
        st.markdown("##### Customer Segmentation")
        
        # Customer value segments
        segments = {
            'High-Value': {'count': 234, 'revenue': 156000, 'color': '#FF6B6B'},
            'Mid-Value': {'count': 567, 'revenue': 89000, 'color': '#4ECDC4'},
            'Low-Value': {'count': 1234, 'revenue': 34000, 'color': '#45B7D1'}
        }
        
        fig = go.Figure()
        
        for segment, data in segments.items():
            fig.add_trace(go.Scatter(
                x=[data['count']],
                y=[data['revenue']],
                mode='markers',
                name=segment,
                marker=dict(
                    size=data['revenue']/2000,
                    color=data['color'],
                    opacity=0.7
                ),
                text=segment,
                textposition="middle center"
            ))
        
        fig.update_layout(
            title='Customer Segments by Value',
            xaxis_title='Customer Count',
            yaxis_title='Revenue Contribution ($)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Focus on mid-value segment conversion")
    
    with col3:
        st.markdown("##### Seasonal Patterns")
        
        # Seasonal revenue analysis
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_factor = [0.85, 0.9, 1.05, 1.1, 1.15, 1.2, 
                          1.25, 1.3, 1.15, 1.1, 1.35, 1.4]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=seasonal_factor,
            mode='lines+markers',
            name='Seasonal Factor',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Seasonal Revenue Patterns',
            xaxis_title='Month',
            yaxis_title='Seasonal Factor',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        peak_month = months[np.argmax(seasonal_factor)]
        st.success(f"💡 Peak season: {peak_month}")

def show_customer_analytics():
    """Display customer analytics and insights"""
    st.subheader("👥 Customer Analytics & Behavior")
    
    # Customer lifecycle analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Customer Lifecycle Analysis")
        
        # Customer journey stages
        journey_stages = {
            'Awareness': 10000,
            'Interest': 5000,
            'Consideration': 2500,
            'Purchase': 1200,
            'Retention': 800,
            'Advocacy': 300
        }
        
        fig = go.Figure(go.Funnel(
            y=list(journey_stages.keys()),
            x=list(journey_stages.values()),
            textinfo="value+percent initial",
            marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ))
        
        fig.update_layout(
            title='Customer Journey Funnel',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Customer Lifetime Value Distribution")
        
        # CLV distribution
        clv_data = st.session_state.ml_models.predict_customer_ltv()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=clv_data,
            nbinsx=25,
            name='CLV Distribution',
            marker_color='lightblue'
        ))
        
        # Add mean line
        mean_clv = np.mean(clv_data)
        fig.add_vline(x=mean_clv, line_dash="dash", line_color="red",
                     annotation_text=f"Mean CLV: ${mean_clv:.0f}")
        
        fig.update_layout(
            title='Customer Lifetime Value Distribution',
            xaxis_title='CLV ($)',
            yaxis_title='Number of Customers',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer segmentation insights
    st.markdown("#### 🎯 Customer Segmentation Insights")
    
    # Generate customer segments
    segment_analysis = st.session_state.ml_models.analyze_customer_segments()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Champions (High Value, High Engagement)")
        st.metric("Customers", f"{segment_analysis['champions']['count']:,}")
        st.metric("Avg CLV", f"${segment_analysis['champions']['avg_clv']:,.0f}")
        st.metric("Retention Rate", f"{segment_analysis['champions']['retention']:.1%}")
        
        st.success("💡 **Strategy**: VIP treatment, exclusive offers")
        
    with col2:
        st.markdown("##### Potential Loyalists (Medium Value, High Engagement)")
        st.metric("Customers", f"{segment_analysis['potential_loyalists']['count']:,}")
        st.metric("Avg CLV", f"${segment_analysis['potential_loyalists']['avg_clv']:,.0f}")
        st.metric("Retention Rate", f"{segment_analysis['potential_loyalists']['retention']:.1%}")
        
        st.warning("💡 **Strategy**: Personalized recommendations")
        
    with col3:
        st.markdown("##### At Risk (High Value, Low Engagement)")
        st.metric("Customers", f"{segment_analysis['at_risk']['count']:,}")
        st.metric("Avg CLV", f"${segment_analysis['at_risk']['avg_clv']:,.0f}")
        st.metric("Retention Rate", f"{segment_analysis['at_risk']['retention']:.1%}")
        
        st.error("💡 **Strategy**: Win-back campaigns")
    
    # Churn prediction insights
    st.markdown("#### 🚨 Churn Prevention Insights")
    
    churn_insights = st.session_state.ml_models.generate_churn_insights()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn risk factors
        st.markdown("##### Top Churn Risk Factors")
        
        risk_factors = churn_insights['risk_factors']
        
        fig = px.bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            title='Churn Risk Factors by Importance',
            color=list(risk_factors.values()),
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Retention strategies effectiveness
        st.markdown("##### Retention Strategy Effectiveness")
        
        strategies = {
            'Personalized Offers': 85,
            'Customer Support': 78,
            'Loyalty Program': 72,
            'Price Discounts': 65,
            'Product Improvements': 82
        }
        
        fig = px.bar(
            x=list(strategies.keys()),
            y=list(strategies.values()),
            title='Retention Strategy Success Rate (%)',
            color=list(strategies.values()),
            color_continuous_scale='greens'
        )
        fig.update_layout(xaxis={'tickangle': 45})
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer satisfaction analysis
    st.markdown("#### 😊 Customer Satisfaction Analysis")
    
    satisfaction_data = st.session_state.data_generator.generate_satisfaction_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # NPS score
        nps_score = satisfaction_data['nps_score']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=nps_score,
            title={'text': "Net Promoter Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-100, 0], 'color': "lightgray"},
                    {'range': [0, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer satisfaction by category
        categories = ['Product Quality', 'Customer Service', 'Pricing', 'Delivery', 'User Experience']
        scores = [np.random.uniform(3.5, 4.8) for _ in categories]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Satisfaction Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )),
            showlegend=True,
            title='Satisfaction by Category',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Customer feedback trends
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        feedback_volume = np.random.poisson(25, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=feedback_volume,
            mode='lines+markers',
            name='Daily Feedback',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='Customer Feedback Volume',
            xaxis_title='Date',
            yaxis_title='Feedback Count',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_market_opportunities():
    """Display market opportunities and competitive analysis"""
    st.subheader("🎯 Market Opportunities & Competitive Intelligence")
    
    # Market analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Market Size & Growth Analysis")
        
        # Market size over time
        years = list(range(2020, 2025))
        market_size = [1.2, 1.45, 1.8, 2.1, 2.5]  # in billions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=market_size,
            mode='lines+markers',
            name='Total Addressable Market',
            line=dict(color='blue', width=4),
            marker=dict(size=10)
        ))
        
        # Add growth trend
        growth_rate = [(market_size[i+1]/market_size[i] - 1) * 100 for i in range(len(market_size)-1)]
        avg_growth = np.mean(growth_rate)
        
        fig.update_layout(
            title=f'Market Size Growth (CAGR: {avg_growth:.1f}%)',
            xaxis_title='Year',
            yaxis_title='Market Size ($B)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Competitive Positioning")
        
        # Competitive landscape
        competitors = {
            'Our Company': {'market_share': 15, 'growth_rate': 25, 'size': 50},
            'Competitor A': {'market_share': 28, 'growth_rate': 12, 'size': 80},
            'Competitor B': {'market_share': 22, 'growth_rate': 8, 'size': 60},
            'Competitor C': {'market_share': 18, 'growth_rate': 15, 'size': 45},
            'Others': {'market_share': 17, 'growth_rate': 5, 'size': 30}
  }

  fig = go.Figure()
        
        for company, data in competitors.items():
            color = '#FF6B6B' if company == 'Our Company' else '#4ECDC4'
            fig.add_trace(go.Scatter(
                x=[data['market_share']],
                y=[data['growth_rate']],
                mode='markers+text',
                name=company,
                marker=dict(
                    size=data['size'],
                    color=color,
                    opacity=0.7
                ),
                text=company,
                textposition="middle center"
            ))
        
        fig.update_layout(
            title='Competitive Positioning Matrix',
            xaxis_title='Market Share (%)',
            yaxis_title='Growth Rate (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic opportunities
    st.markdown("#### 🌍 Geographic Expansion Opportunities")
    
    # Generate geographic data
    geo_opportunities = st.session_state.ml_models.analyze_geographic_opportunities()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### High Potential Markets")
        
        for market in geo_opportunities['high_potential']:
            st.write(f"🟢 **{market['region']}**")
            st.write(f"   • Market Size: ${market['size']}M")
            st.write(f"   • Growth Rate: {market['growth']}%")
            st.write(f"   • Competition: {market['competition']}")
            st.write("")
    
    with col2:
        st.markdown("##### Medium Potential Markets")
        
        for market in geo_opportunities['medium_potential']:
            st.write(f"🟡 **{market['region']}**")
            st.write(f"   • Market Size: ${market['size']}M")
            st.write(f"   • Growth Rate: {market['growth']}%")
            st.write(f"   • Competition: {market['competition']}")
            st.write("")
    
    with col3:
        st.markdown("##### Entry Barriers & Risks")
        
        barriers = [
            "Regulatory compliance requirements",
            "Cultural adaptation needs",
            "Local partnership requirements",
            "Currency exchange risks",
            "Supply chain complexity"
        ]
        
        for barrier in barriers:
            st.write(f"⚠️ {barrier}")
    
    # Product opportunities
    st.markdown("#### 🚀 Product Development Opportunities")
    
    product_opportunities = st.session_state.ml_models.identify_product_opportunities()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Gap Analysis")
        
        # Feature gap analysis
        features = ['Price', 'Quality', 'Features', 'Support', 'Brand']
        our_score = [7.5, 8.2, 7.8, 8.5, 7.3]
        competitor_avg = [7.8, 7.9, 8.3, 7.6, 8.1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=our_score,
            theta=features,
            fill='toself',
            name='Our Company',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=competitor_avg,
            theta=features,
            fill='toself',
            name='Competitor Average',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title='Competitive Gap Analysis',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Innovation Opportunities")
        
        innovations = product_opportunities['innovations']
        
        for innovation in innovations:
            with st.expander(f"💡 {innovation['title']}"):
                st.write(f"**Description:** {innovation['description']}")
                st.write(f"**Market Demand:** {innovation['demand']}")
                st.write(f"**Development Effort:** {innovation['effort']}")
                st.write(f"**Potential ROI:** {innovation['roi']}")
                
                # ROI vs Effort scatter
                effort_score = {'Low': 1, 'Medium': 2, 'High': 3}[innovation['effort']]
                roi_score = int(innovation['roi'].replace('%', ''))
                
                if roi_score > 50 and effort_score <= 2:
                    st.success("🚀 High priority opportunity")
                elif roi_score > 30:
                    st.info("📈 Consider for development")
                else:
                    st.warning("⏳ Monitor market conditions")

def show_strategic_recommendations():
    """Display AI-generated strategic recommendations"""
    st.subheader("🚀 Strategic Recommendations Engine")
    
    # Priority recommendations
    st.markdown("#### 🎯 Priority Actions (Next 90 Days)")
    
    priority_actions = [
        {
            "title": "Implement Dynamic Pricing Strategy",
            "description": "ML models show 15-23% revenue increase potential through dynamic pricing optimization",
            "impact": "High",
            "effort": "Medium",
            "timeline": "6-8 weeks",
            "roi": "185%",
            "risk": "Low"
        },
        {
            "title": "Launch Targeted Retention Campaign",
            "description": "342 high-value customers identified at churn risk - immediate intervention recommended",
            "impact": "Critical",
            "effort": "Low",
            "timeline": "2-3 weeks",
            "roi": "240%",
            "risk": "Very Low"
        },
        {
            "title": "Expand Geographic Presence",
            "description": "Southeast Asia markets show 45% growth potential with low competition",
            "impact": "High",
            "effort": "High",
            "timeline": "12-16 weeks",
            "roi": "156%",
            "risk": "Medium"
        }
    ]
    
    for i, action in enumerate(priority_actions):
        with st.expander(f"🔥 Priority {i+1}: {action['title']}", expanded=i==0):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(action['description'])
                
                # Progress visualization
                st.markdown("**Implementation Roadmap:**")
                if action['timeline'] == "2-3 weeks":
                    st.progress(0.8)
                    st.write("📅 Week 1-2: Planning & Setup")
                    st.write("📅 Week 3: Launch & Monitor")
                elif action['timeline'] == "6-8 weeks":
                    st.progress(0.3)
                    st.write("📅 Week 1-2: Analysis & Strategy")
                    st.write("📅 Week 3-6: Development & Testing")
                    st.write("📅 Week 7-8: Rollout & Optimization")
                else:
                    st.progress(0.1)
                    st.write("📅 Month 1-2: Market Research & Planning")
                    st.write("📅 Month 3: Partnership & Setup")
                    st.write("📅 Month 4: Launch & Scale")
            
            with col2:
                # Metrics
                st.metric("Expected ROI", action['roi'])
                st.metric("Timeline", action['timeline'])
                
                # Impact indicator
                if action['impact'] == 'Critical':
                    st.error(f"🔴 {action['impact']} Impact")
                elif action['impact'] == 'High':
                    st.warning(f"🟡 {action['impact']} Impact")
                else:
                    st.info(f"🔵 {action['impact']} Impact")
                
                # Risk assessment
                risk_color = {
                    'Very Low': 'success',
                    'Low': 'success',
                    'Medium': 'warning',
                    'High': 'error'
                }
                
                if action['risk'] in ['Very Low', 'Low']:
                    st.success(f"✅ {action['risk']} Risk")
                elif action['risk'] == 'Medium':
                    st.warning(f"⚠️ {action['risk']} Risk")
                else:
                    st.error(f"🚨 {action['risk']} Risk")
    
    # Investment recommendations
    st.markdown("#### 💰 Investment Allocation Recommendations")
    
    investment_areas = {
        'Customer Retention': {'allocation': 35, 'expected_return': 240},
        'Product Development': {'allocation': 25, 'expected_return': 180},
        'Market Expansion': {'allocation': 20, 'expected_return': 156},
        'Technology Infrastructure': {'allocation': 15, 'expected_return': 125},
        'Marketing & Acquisition': {'allocation': 5, 'expected_return': 95}
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Investment allocation pie chart
        fig = px.pie(
            values=[data['allocation'] for data in investment_areas.values()],
            names=list(investment_areas.keys()),
            title='Recommended Investment Allocation (%)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROI comparison
        areas = list(investment_areas.keys())
        rois = [data['expected_return'] for data in investment_areas.values()]
        
        fig = px.bar(
            x=rois,
            y=areas,
            orientation='h',
            title='Expected ROI by Investment Area (%)',
            color=rois,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Long-term strategic vision
    st.markdown("#### 🔮 Long-term Strategic Vision (12-18 months)")
    
    strategic_initiatives = [
        {
            "theme": "AI-First Organization",
            "description": "Transform into an AI-native company with automated decision-making across all business functions",
            "key_milestones": [
                "Implement ML-driven pricing across all products",
                "Deploy predictive analytics for demand forecasting",
                "Automate customer service with AI chatbots",
                "Launch personalization engine for all customer touchpoints"
            ]
        },
        {
            "theme": "Global Market Leadership",
            "description": "Establish dominant position in 3 new geographic markets and 2 adjacent product categories",
            "key_milestones": [
                "Enter Southeast Asian markets",
                "Launch complementary product line",
                "Establish strategic partnerships in target regions",
                "Achieve 25% international revenue mix"
            ]
        },
        {
            "theme": "Customer-Centric Innovation",
            "description": "Build world-class customer experience through data-driven insights and personalization",
            "key_milestones": [
                "Achieve NPS score above 70",
                "Reduce churn rate below 3%",
                "Increase customer lifetime value by 40%",
                "Launch customer co-innovation program"
            ]
        }
    ]
    
    for initiative in strategic_initiatives:
        with st.expander(f"🎯 {initiative['theme']}"):
            st.write(initiative['description'])
            
            st.markdown("**Key Milestones:**")
            for milestone in initiative['key_milestones']:
                st.write(f"• {milestone}")
    
    # Resource requirements
    st.markdown("#### 📊 Resource Requirements Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Human Resources")
        
        hr_needs = {
            'Data Scientists': 3,
            'ML Engineers': 2,
            'Product Managers': 2,
            'Sales Professionals': 5,
            'Customer Success': 4
        }
        
        for role, count in hr_needs.items():
            st.write(f"• {role}: {count} positions")
        
        st.metric("Total New Hires", sum(hr_needs.values()))
    
    with col2:
        st.markdown("##### Technology Investment")
        
        tech_investments = {
            'ML Platform': '$150K',
            'Data Infrastructure': '$200K',
            'Analytics Tools': '$75K',
            'Security & Compliance': '$100K',
            'Integration Costs': '$50K'
        }
        
        for item, cost in tech_investments.items():
            st.write(f"• {item}: {cost}")
        
        total_tech = sum([int(cost.replace('$', '').replace('K', '')) for cost in tech_investments.values()])
        st.metric("Total Technology", f"${total_tech}K")
    
    with col3:
        st.markdown("##### Expected Outcomes")
        
        outcomes = [
            "Revenue increase: 35-45%",
            "Cost reduction: 15-20%",
            "Customer satisfaction: +25%",
            "Market share growth: +8%",
            "Operational efficiency: +30%"
        ]
        
        for outcome in outcomes:
            st.write(f"✅ {outcome}")

if __name__ == "__main__":
    main()
          
