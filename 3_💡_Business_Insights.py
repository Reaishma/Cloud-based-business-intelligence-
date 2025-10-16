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

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide")

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'metrics_calc' not in st.session_state:
    st.session_state.metrics_calc = MetricsCalculator()

st.title("💡 Business Insights & Analytics")
st.markdown("### AI-Powered Strategic Intelligence")
st.markdown("---")

# Insight Categories
insight_type = st.selectbox(
    "Select Insight Category",
    [
        "Revenue Forecasting",
        "Customer Analytics",
        "Churn Analysis",
        "Market Opportunities",
        "Product Innovation",
        "Operational Efficiency"
    ]
)

st.markdown("---")

# Revenue Forecasting Insights
if insight_type == "Revenue Forecasting":
    st.subheader("📈 Revenue Forecasting & Trends")
    
    forecast_data = st.session_state.ml_models.generate_revenue_forecast()
    
    col1, col2, col3 = st.columns(3)
    
    current_revenue = forecast_data['historical']['revenue'][-1]
    forecasted_revenue = forecast_data['forecast']['revenue'][-1]
    growth = ((forecasted_revenue - current_revenue) / current_revenue) * 100
    
    with col1:
        st.metric("Current Monthly Revenue", f"${current_revenue:,.0f}")
    with col2:
        st.metric("6-Month Forecast", f"${forecasted_revenue:,.0f}", f"{growth:.1f}%")
    with col3:
        confidence = np.random.uniform(85, 95)
        st.metric("Forecast Confidence", f"{confidence:.1f}%")
    
    # Revenue forecast chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_data['historical']['dates'],
        y=forecast_data['historical']['revenue'],
        mode='lines',
        name='Historical Revenue',
        line=dict(color='#4ECDC4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['forecast']['dates'],
        y=forecast_data['forecast']['revenue'],
        mode='lines',
        name='Forecasted Revenue',
        line=dict(color='#FF6B6B', width=3, dash='dash')
    ))
    
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
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    
    fig.update_layout(
        title='Revenue Forecast (6 Months)',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🔍 Key Revenue Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📊 **Trend Analysis**")
        st.write("• Revenue shows consistent upward trend")
        st.write("• Seasonal patterns indicate Q4 peak performance")
        st.write("• Year-over-year growth projected at 18.5%")
        
        st.success("💰 **Growth Drivers**")
        st.write("• New customer acquisition: +25%")
        st.write("• Average order value increase: +12%")
        st.write("• Customer retention improvement: +8%")
    
    with col2:
        st.warning("⚠️ **Risk Factors**")
        st.write("• Market saturation in primary segment")
        st.write("• Increased competition pressure")
        st.write("• Economic uncertainty impact")
        
        st.info("🎯 **Recommendations**")
        st.write("• Expand into adjacent markets")
        st.write("• Invest in customer retention programs")
        st.write("• Optimize pricing strategy")

# Customer Analytics Insights
elif insight_type == "Customer Analytics":
    st.subheader("👥 Customer Analytics & Segmentation")
    
    segments = st.session_state.ml_models.analyze_customer_segments()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", f"{segments['champions']['count'] + segments['potential_loyalists']['count'] + segments['at_risk']['count']:,}")
    with col2:
        avg_clv = (segments['champions']['avg_clv'] + segments['potential_loyalists']['avg_clv'] + segments['at_risk']['avg_clv']) / 3
        st.metric("Avg Customer LTV", f"${avg_clv:,.0f}")
    with col3:
        avg_retention = (segments['champions']['retention'] + segments['potential_loyalists']['retention'] + segments['at_risk']['retention']) / 3
        st.metric("Avg Retention Rate", f"{avg_retention:.1%}")
    
    # Segment distribution
    segment_data = pd.DataFrame({
        'Segment': ['Champions', 'Potential Loyalists', 'At Risk'],
        'Count': [segments['champions']['count'], segments['potential_loyalists']['count'], segments['at_risk']['count']],
        'Avg CLV': [segments['champions']['avg_clv'], segments['potential_loyalists']['avg_clv'], segments['at_risk']['avg_clv']],
        'Retention': [segments['champions']['retention'], segments['potential_loyalists']['retention'], segments['at_risk']['retention']]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            segment_data,
            values='Count',
            names='Segment',
            title='Customer Segment Distribution',
            color='Segment',
            color_discrete_map={'Champions': '#6BCF7F', 'Potential Loyalists': '#FFD93D', 'At Risk': '#FF6B6B'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segment_data,
            x='Segment',
            y='Avg CLV',
            title='Average Customer Lifetime Value by Segment',
            color='Avg CLV',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer insights
    st.subheader("💎 Segment Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("🏆 **Champions**")
        st.write(f"• Count: {segments['champions']['count']:,} customers")
        st.write(f"• Avg CLV: ${segments['champions']['avg_clv']:,.0f}")
        st.write(f"• Retention: {segments['champions']['retention']:.1%}")
        st.write("• Strategy: VIP programs, exclusive offers")
    
    with col2:
        st.info("⭐ **Potential Loyalists**")
        st.write(f"• Count: {segments['potential_loyalists']['count']:,} customers")
        st.write(f"• Avg CLV: ${segments['potential_loyalists']['avg_clv']:,.0f}")
        st.write(f"• Retention: {segments['potential_loyalists']['retention']:.1%}")
        st.write("• Strategy: Engagement campaigns, loyalty rewards")
    
    with col3:
        st.warning("⚠️ **At Risk**")
        st.write(f"• Count: {segments['at_risk']['count']:,} customers")
        st.write(f"• Avg CLV: ${segments['at_risk']['avg_clv']:,.0f}")
        st.write(f"• Retention: {segments['at_risk']['retention']:.1%}")
        st.write("• Strategy: Win-back campaigns, personalized offers")

# Churn Analysis Insights
elif insight_type == "Churn Analysis":
    st.subheader("🚨 Churn Risk Analysis & Prevention")
    
    churn_insights = st.session_state.ml_models.generate_churn_insights()
    churn_data = st.session_state.ml_models.predict_churn_risk()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("High Risk", churn_data['high_risk'], "-12")
    with col2:
        st.metric("Medium Risk", churn_data['medium_risk'], "+5")
    with col3:
        st.metric("Low Risk", churn_data['low_risk'], "+18")
    with col4:
        churn_rate = (churn_data['high_risk'] / churn_data['total_customers']) * 100
        st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
    
    # Risk factor analysis
    st.subheader("🔍 Top Churn Risk Factors")
    
    risk_df = pd.DataFrame({
        'Risk Factor': list(churn_insights['risk_factors'].keys()),
        'Impact': list(churn_insights['risk_factors'].values())
    }).sort_values('Impact', ascending=True)
    
    fig = px.bar(
        risk_df,
        x='Impact',
        y='Risk Factor',
        orientation='h',
        title='Churn Risk Factor Importance',
        color='Impact',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Churn prevention strategies
    st.subheader("🛡️ Churn Prevention Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📞 **Immediate Actions**")
        st.write("• Proactive outreach to high-risk customers")
        st.write("• Personalized retention offers")
        st.write("• Priority customer service")
        st.write("• Product usage training")
        
        st.success("💰 **Expected Impact**")
        potential_revenue_saved = churn_data['high_risk'] * np.random.uniform(800, 1500)
        st.write(f"• Potential revenue saved: ${potential_revenue_saved:,.0f}")
        st.write(f"• Customers retained: {int(churn_data['high_risk'] * 0.65):,}")
        st.write(f"• ROI on retention spend: 3.2x")
    
    with col2:
        st.warning("📊 **Long-term Initiatives**")
        st.write("• Improve onboarding experience")
        st.write("• Enhance product features based on feedback")
        st.write("• Build customer community")
        st.write("• Implement loyalty program")
        
        st.info("📈 **Success Metrics**")
        st.write("• Target churn reduction: 25%")
        st.write("• Timeline: 6 months")
        st.write("• Investment required: $150K")
        st.write("• Projected annual savings: $480K")

# Market Opportunities Insights
elif insight_type == "Market Opportunities":
    st.subheader("🌍 Market Expansion Opportunities")
    
    geo_opportunities = st.session_state.ml_models.analyze_geographic_opportunities()
    
    st.subheader("🎯 High-Potential Markets")
    
    for region in geo_opportunities['high_potential']:
        with st.expander(f"🌟 {region['region']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Market Size", f"${region['size']}M")
            with col2:
                st.metric("Growth Rate", f"{region['growth']:.1f}%")
            with col3:
                st.metric("Competition", region['competition'])
            
            st.info("**Opportunity Analysis:**")
            st.write(f"• Untapped market with {region['growth']:.1f}% annual growth")
            st.write(f"• {region['competition']} competition allows for strong market entry")
            st.write(f"• Estimated 3-year revenue potential: ${region['size'] * 0.15:.0f}M")
            st.write(f"• Recommended entry strategy: Strategic partnerships + digital marketing")
    
    st.subheader("💼 Medium-Potential Markets")
    
    col1, col2 = st.columns(2)
    
    for idx, region in enumerate(geo_opportunities['medium_potential']):
        with [col1, col2][idx]:
            st.write(f"**{region['region']}**")
            st.write(f"• Market Size: ${region['size']}M")
            st.write(f"• Growth: {region['growth']:.1f}%")
            st.write(f"• Competition: {region['competition']}")
    
    # Market entry strategy
    st.markdown("---")
    st.subheader("🚀 Recommended Market Entry Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Phase 1: Market Research (Months 1-2)**")
        st.write("• Conduct detailed market analysis")
        st.write("• Identify local partners")
        st.write("• Assess regulatory requirements")
        st.write("• Budget: $50K")
        
        st.success("**Phase 2: Pilot Launch (Months 3-5)**")
        st.write("• Limited product offering")
        st.write("• Partner with local distributors")
        st.write("• Digital marketing campaign")
        st.write("• Budget: $200K")
    
    with col2:
        st.success("**Phase 3: Scale Operations (Months 6-12)**")
        st.write("• Expand product portfolio")
        st.write("• Build local team")
        st.write("• Establish distribution network")
        st.write("• Budget: $500K")
        
        st.info("**Expected Outcomes**")
        st.write("• Year 1 Revenue: $2.5M")
        st.write("• Market Share: 8-12%")
        st.write("• ROI: 180%")
        st.write("• Break-even: Month 9")

# Product Innovation Insights
elif insight_type == "Product Innovation":
    st.subheader("🚀 Product Innovation Opportunities")
    
    innovations = st.session_state.ml_models.identify_product_opportunities()
    
    for innovation in innovations['innovations']:
        with st.expander(f"💡 {innovation['title']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Customer Demand", innovation['demand'])
            with col2:
                st.metric("Development Effort", innovation['effort'])
            with col3:
                st.metric("Expected ROI", innovation['roi'])
            with col4:
                priority_score = {"High": 95, "Medium": 70, "Low": 40}[innovation['demand']]
                st.metric("Priority Score", f"{priority_score}/100")
            
            st.write(f"**Description:** {innovation['description']}")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.info("**Business Impact:**")
                st.write("• Revenue increase: 15-25%")
                st.write("• Customer satisfaction: +18%")
                st.write("• Market differentiation: High")
                st.write("• Time to market: 4-6 months")
            
            with col_b:
                st.success("**Implementation Plan:**")
                st.write("• MVP development: 2 months")
                st.write("• Beta testing: 1 month")
                st.write("• Full rollout: 1 month")
                st.write("• Investment required: $250K")
    
    # Innovation roadmap
    st.markdown("---")
    st.subheader("📅 Innovation Roadmap")
    
    roadmap_data = pd.DataFrame({
        'Quarter': ['Q1 2025', 'Q2 2025', 'Q3 2025', 'Q4 2025'],
        'Initiative': [
            innovations['innovations'][0]['title'],
            innovations['innovations'][1]['title'],
            innovations['innovations'][2]['title'],
            'Platform Enhancement'
        ],
        'Investment': [250, 300, 200, 150],
        'Expected ROI': [180, 145, 220, 160]
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Quarterly Investment', 'Expected ROI'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=roadmap_data['Quarter'], y=roadmap_data['Investment'], name='Investment ($K)', marker_color='#FF6B6B'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=roadmap_data['Quarter'], y=roadmap_data['Expected ROI'], name='ROI (%)', marker_color='#4ECDC4'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Operational Efficiency Insights
elif insight_type == "Operational Efficiency":
    st.subheader("⚙️ Operational Efficiency Analysis")
    
    operational_data = st.session_state.data_generator.generate_operational_data()
    financial_data = st.session_state.data_generator.generate_financial_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Order Fulfillment", f"{operational_data['order_fulfillment_time']:.1f} days", "-0.3 days")
    with col2:
        st.metric("Inventory Turnover", f"{operational_data['inventory_turnover']:.1f}x", "+1.2x")
    with col3:
        st.metric("Supply Chain Efficiency", f"{operational_data['supply_chain_efficiency']:.1f}%", "+2.5%")
    with col4:
        st.metric("System Uptime", f"{operational_data['system_uptime']:.2f}%", "+0.15%")
    
    # Efficiency metrics
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics_data = pd.DataFrame({
            'Metric': ['Response Time', 'Employee Productivity', 'Inventory Turnover', 'Supply Chain Eff.'],
            'Current': [
                operational_data['response_time'],
                operational_data['employee_productivity'],
                operational_data['inventory_turnover'],
                operational_data['supply_chain_efficiency']
            ],
            'Target': [250, 115, 18, 98],
            'Industry Avg': [350, 100, 12, 90]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Current', x=metrics_data['Metric'], y=metrics_data['Current'], marker_color='#4ECDC4'))
        fig.add_trace(go.Bar(name='Target', x=metrics_data['Metric'], y=metrics_data['Target'], marker_color='#FFD93D'))
        fig.add_trace(go.Bar(name='Industry Avg', x=metrics_data['Metric'], y=metrics_data['Industry Avg'], marker_color='#95A5A6'))
        
        fig.update_layout(title='Operational Metrics Comparison', barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.info("**🎯 Optimization Opportunities**")
        st.write("")
        st.write("**Process Automation:**")
        st.write("• Potential time savings: 35 hours/week")
        st.write("• Cost reduction: $180K annually")
        st.write("• Implementation timeline: 3 months")
        st.write("")
        st.write("**Inventory Optimization:**")
        st.write("• Reduce carrying costs: 15%")
        st.write("• Improve stock turnover: 25%")
        st.write("• Free up capital: $500K")
        st.write("")
        st.write("**Supply Chain Enhancement:**")
        st.write("• Reduce lead times: 20%")
        st.write("• Lower logistics costs: 12%")
        st.write("• Annual savings: $320K")
    
    # Financial impact
    st.markdown("---")
    st.subheader("💰 Financial Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Revenue Growth", f"{financial_data['revenue_growth']:.1f}%")
        st.metric("Profit Margin", f"{financial_data['profit_margin']:.1f}%")
    
    with col2:
        st.metric("CAC", f"${financial_data['customer_acquisition_cost']:.0f}")
        st.metric("LTV", f"${financial_data['lifetime_value']:.0f}")
    
    with col3:
        ltv_cac_ratio = financial_data['lifetime_value'] / financial_data['customer_acquisition_cost']
        st.metric("LTV:CAC Ratio", f"{ltv_cac_ratio:.1f}:1")
        st.metric("MRR", f"${financial_data['monthly_recurring_revenue']:.0f}K")
    
    # Recommendations
    st.subheader("📋 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Quick Wins (0-3 months)**")
        st.write("1. Automate order processing workflows")
        st.write("2. Implement predictive inventory management")
        st.write("3. Optimize supplier negotiations")
        st.write("4. Streamline approval processes")
        st.write("• Expected savings: $85K")
        st.write("• Implementation cost: $25K")
    
    with col2:
        st.info("**Long-term Initiatives (3-12 months)**")
        st.write("1. Deploy AI-powered demand forecasting")
        st.write("2. Restructure distribution network")
        st.write("3. Implement advanced analytics platform")
        st.write("4. Launch continuous improvement program")
        st.write("• Expected savings: $450K annually")
        st.write("• Implementation cost: $150K")

# Executive Summary
st.markdown("---")
st.subheader("📄 Executive Summary")

col1, col2 = st.columns(2)

with col1:
    st.info("**Key Findings:**")
    st.write("• Revenue growth trajectory remains strong at 18.5% YoY")
    st.write("• Customer retention opportunities worth $480K annually")
    st.write("• Market expansion could generate $2.5M in Year 1")
    st.write("• Product innovations show 180-220% ROI potential")
    st.write("• Operational efficiencies could save $450K annually")

with col2:
    st.success("**Strategic Priorities:**")
    st.write("1. **Immediate:** Launch churn prevention campaign")
    st.write("2. **Short-term:** Develop AI personalization engine")
    st.write("3. **Medium-term:** Expand into Southeast Asia")
    st.write("4. **Long-term:** Build subscription service model")
    st.write("• Total investment: $1.2M | Expected return: $4.5M")
