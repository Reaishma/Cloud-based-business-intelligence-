-- Advanced Analytics Queries for Cloud BI Platform
-- Business Intelligence and Machine Learning Support Queries

-- =====================================
-- SALES ANALYTICS QUERIES
-- =====================================

-- 1. Sales Performance Dashboard
-- Monthly sales trends with year-over-year comparison
WITH monthly_sales AS (
    SELECT 
        d.year,
        d.month,
        d.month_name,
        SUM(s.total_amount) as monthly_revenue,
        COUNT(s.sale_key) as transaction_count,
        AVG(s.total_amount) as avg_transaction_value
    FROM fact_sales s
    JOIN dim_date d ON s.date_key = d.date_key
    GROUP BY d.year, d.month, d.month_name
),
yoy_comparison AS (
    SELECT 
        *,
        LAG(monthly_revenue, 12) OVER (ORDER BY year, month) as prev_year_revenue,
        (monthly_revenue - LAG(monthly_revenue, 12) OVER (ORDER BY year, month)) / 
        NULLIF(LAG(monthly_revenue, 12) OVER (ORDER BY year, month), 0) * 100 as yoy_growth_rate
    FROM monthly_sales
)
SELECT 
    year,
    month_name,
    monthly_revenue,
    transaction_count,
    avg_transaction_value,
    prev_year_revenue,
    ROUND(yoy_growth_rate, 2) as yoy_growth_percentage
FROM yoy_comparison
WHERE year >= EXTRACT(YEAR FROM CURRENT_DATE) - 2
ORDER BY year, month;

-- 2. Product Performance Analysis
-- Top performing products with profit margin analysis
SELECT 
    p.product_name,
    p.category,
    p.brand,
    COUNT(s.sale_key) as total_orders,
    SUM(s.quantity) as units_sold,
    SUM(s.total_amount) as total_revenue,
    SUM(s.profit_amount) as total_profit,
    ROUND(SUM(s.profit_amount) / NULLIF(SUM(s.total_amount), 0) * 100, 2) as profit_margin_pct,
    ROUND(AVG(s.total_amount), 2) as avg_order_value,
    RANK() OVER (ORDER BY SUM(s.total_amount) DESC) as revenue_rank
FROM fact_sales s
JOIN dim_products p ON s.product_key = p.product_key
JOIN dim_date d ON s.date_key = d.date_key
WHERE d.full_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY p.product_id, p.product_name, p.category, p.brand
ORDER BY total_revenue DESC
LIMIT 50;

-- 3. Regional Sales Performance
-- Sales performance by geography with growth metrics
SELECT 
    g.region,
    g.country,
    COUNT(DISTINCT s.customer_key) as unique_customers,
    COUNT(s.sale_key) as total_transactions,
    SUM(s.total_amount) as total_revenue,
    ROUND(AVG(s.total_amount), 2) as avg_transaction_value,
    ROUND(SUM(s.total_amount) / COUNT(DISTINCT s.customer_key), 2) as revenue_per_customer,
    ROUND(
        (SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN s.total_amount ELSE 0 END) -
         SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '60 days' 
                   AND d.full_date < CURRENT_DATE - INTERVAL '30 days' THEN s.total_amount ELSE 0 END)) /
        NULLIF(SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '60 days' 
                         AND d.full_date < CURRENT_DATE - INTERVAL '30 days' THEN s.total_amount ELSE 0 END), 0) * 100, 2
    ) as month_over_month_growth
FROM fact_sales s
JOIN dim_geography g ON s.geography_key = g.geography_key
JOIN dim_date d ON s.date_key = d.date_key
WHERE d.full_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY g.region, g.country
ORDER BY total_revenue DESC;

-- =====================================
-- CUSTOMER ANALYTICS QUERIES
-- =====================================

-- 4. Customer Segmentation Analysis (RFM)
-- Recency, Frequency, Monetary analysis for customer segmentation
WITH customer_rfm AS (
    SELECT 
        s.customer_key,
        c.customer_id,
        c.first_name,
        c.last_name,
        c.customer_segment,
        CURRENT_DATE - MAX(d.full_date) as recency_days,
        COUNT(s.sale_key) as frequency,
        SUM(s.total_amount) as monetary_value,
        AVG(s.total_amount) as avg_order_value
    FROM fact_sales s
    JOIN dim_customers c ON s.customer_key = c.customer_key
    JOIN dim_date d ON s.date_key = d.date_key
    GROUP BY s.customer_key, c.customer_id, c.first_name, c.last_name, c.customer_segment
),
rfm_scores AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency_days) as recency_score,
        NTILE(5) OVER (ORDER BY frequency DESC) as frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value DESC) as monetary_score
    FROM customer_rfm
),
rfm_segments AS (
    SELECT 
        *,
        CASE 
            WHEN frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
            WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
            WHEN recency_score >= 4 AND monetary_score >= 4 THEN 'Potential Loyalists'
            WHEN recency_score >= 4 AND frequency_score <= 2 THEN 'New Customers'
            WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
            WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Cannot Lose Them'
            WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'Hibernating'
            ELSE 'Others'
        END as rfm_segment
    FROM rfm_scores
)
SELECT 
    rfm_segment,
    COUNT(*) as customer_count,
    ROUND(AVG(recency_days), 1) as avg_recency_days,
    ROUND(AVG(frequency), 1) as avg_frequency,
    ROUND(AVG(monetary_value), 2) as avg_monetary_value,
    ROUND(SUM(monetary_value), 2) as total_revenue,
    ROUND(AVG(avg_order_value), 2) as avg_order_value
FROM rfm_segments
GROUP BY rfm_segment
ORDER BY total_revenue DESC;

-- 5. Customer Lifetime Value Analysis
-- CLV calculation and customer value distribution
WITH customer_metrics AS (
    SELECT 
        s.customer_key,
        c.customer_id,
        c.first_name,
        c.last_name,
        c.registration_date,
        MIN(d.full_date) as first_purchase_date,
        MAX(d.full_date) as last_purchase_date,
        COUNT(s.sale_key) as total_orders,
        SUM(s.total_amount) as total_spent,
        AVG(s.total_amount) as avg_order_value,
        COUNT(s.sale_key) * 1.0 / NULLIF(
            EXTRACT(DAYS FROM (MAX(d.full_date) - MIN(d.full_date))) / 30.0, 0
        ) as monthly_purchase_frequency
    FROM fact_sales s
    JOIN dim_customers c ON s.customer_key = c.customer_key
    JOIN dim_date d ON s.date_key = d.date_key
    GROUP BY s.customer_key, c.customer_id, c.first_name, c.last_name, c.registration_date
),
clv_calculation AS (
    SELECT 
        *,
        CASE 
            WHEN monthly_purchase_frequency > 0 
            THEN avg_order_value * monthly_purchase_frequency * 24 -- 24 month projection
            ELSE total_spent
        END as calculated_clv,
        CASE 
            WHEN CURRENT_DATE - last_purchase_date > 90 THEN 'High Risk'
            WHEN CURRENT_DATE - last_purchase_date > 60 THEN 'Medium Risk'
            ELSE 'Active'
        END as churn_risk
    FROM customer_metrics
)
SELECT 
    customer_id,
    first_name,
    last_name,
    total_orders,
    ROUND(total_spent, 2) as total_spent,
    ROUND(avg_order_value, 2) as avg_order_value,
    ROUND(calculated_clv, 2) as predicted_clv,
    churn_risk,
    NTILE(10) OVER (ORDER BY calculated_clv DESC) as clv_decile
FROM clv_calculation
ORDER BY calculated_clv DESC;

-- =====================================
-- CHURN PREDICTION SUPPORT QUERIES
-- =====================================

-- 6. Churn Risk Indicators
-- Identify customers at risk of churning based on behavioral patterns
WITH customer_behavior AS (
    SELECT 
        s.customer_key,
        c.customer_id,
        c.first_name,
        c.last_name,
        c.customer_segment,
        CURRENT_DATE - MAX(d.full_date) as days_since_last_purchase,
        COUNT(s.sale_key) as total_orders,
        SUM(s.total_amount) as total_spent,
        AVG(s.total_amount) as avg_order_value,
        
        -- Recent activity (last 30 days)
        COUNT(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as orders_last_30_days,
        SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN s.total_amount ELSE 0 END) as revenue_last_30_days,
        
        -- Trend analysis (comparing last 30 vs previous 30 days)
        COUNT(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '60 days' 
                    AND d.full_date < CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as orders_prev_30_days,
        SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '60 days' 
                  AND d.full_date < CURRENT_DATE - INTERVAL '30 days' THEN s.total_amount ELSE 0 END) as revenue_prev_30_days
        
    FROM fact_sales s
    JOIN dim_customers c ON s.customer_key = c.customer_key
    JOIN dim_date d ON s.date_key = d.date_key
    GROUP BY s.customer_key, c.customer_id, c.first_name, c.last_name, c.customer_segment
),
churn_risk_scoring AS (
    SELECT 
        *,
        CASE 
            WHEN days_since_last_purchase > 90 THEN 4
            WHEN days_since_last_purchase > 60 THEN 3
            WHEN days_since_last_purchase > 30 THEN 2
            ELSE 1
        END as recency_risk_score,
        
        CASE 
            WHEN orders_last_30_days = 0 AND orders_prev_30_days > 0 THEN 3
            WHEN orders_last_30_days < orders_prev_30_days THEN 2
            ELSE 1
        END as frequency_risk_score,
        
        CASE 
            WHEN revenue_last_30_days = 0 AND revenue_prev_30_days > 0 THEN 3
            WHEN revenue_last_30_days < revenue_prev_30_days * 0.5 THEN 2
            ELSE 1
        END as monetary_risk_score
        
    FROM customer_behavior
    WHERE total_orders >= 2 -- Focus on customers with purchase history
)
SELECT 
    customer_id,
    first_name,
    last_name,
    customer_segment,
    days_since_last_purchase,
    total_orders,
    ROUND(total_spent, 2) as total_spent,
    orders_last_30_days,
    orders_prev_30_days,
    recency_risk_score + frequency_risk_score + monetary_risk_score as total_risk_score,
    CASE 
        WHEN recency_risk_score + frequency_risk_score + monetary_risk_score >= 7 THEN 'High Risk'
        WHEN recency_risk_score + frequency_risk_score + monetary_risk_score >= 5 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as churn_risk_category
FROM churn_risk_scoring
WHERE recency_risk_score + frequency_risk_score + monetary_risk_score >= 4
ORDER BY total_risk_score DESC, total_spent DESC;

-- =====================================
-- INVENTORY AND SUPPLY CHAIN ANALYTICS
-- =====================================

-- 7. Inventory Performance and Optimization
-- Inventory turnover, stockout risk, and reorder recommendations
WITH inventory_metrics AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.category,
        p.inventory_count,
        p.reorder_level,
        p.unit_cost,
        p.unit_price,
        
        -- Sales velocity (last 30 days)
        COUNT(s.sale_key) as orders_last_30_days,
        SUM(s.quantity) as units_sold_last_30_days,
        SUM(s.total_amount) as revenue_last_30_days,
        
        -- Sales velocity (last 90 days)
        COUNT(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as orders_last_90_days,
        SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '90 days' THEN s.quantity ELSE 0 END) as units_sold_last_90_days
        
    FROM dim_products p
    LEFT JOIN fact_sales s ON p.product_key = s.product_key
    LEFT JOIN dim_date d ON s.date_key = d.date_key AND d.full_date >= CURRENT_DATE - INTERVAL '90 days'
    WHERE p.status = 'Active'
    GROUP BY p.product_id, p.product_name, p.category, p.inventory_count, p.reorder_level, p.unit_cost, p.unit_price
),
inventory_analysis AS (
    SELECT 
        *,
        CASE 
            WHEN units_sold_last_30_days > 0 
            THEN inventory_count * 1.0 / (units_sold_last_30_days / 30.0)
            ELSE 999
        END as days_of_inventory,
        
        CASE 
            WHEN units_sold_last_90_days > 0 
            THEN (units_sold_last_90_days / 90.0) * 30 -- Monthly demand forecast
            ELSE 0
        END as forecasted_monthly_demand,
        
        CASE 
            WHEN inventory_count <= reorder_level THEN 'Immediate Reorder'
            WHEN units_sold_last_30_days > 0 AND inventory_count <= (units_sold_last_30_days * 1.5) THEN 'Monitor Closely'
            ELSE 'Normal'
        END as inventory_status
        
    FROM inventory_metrics
)
SELECT 
    product_id,
    product_name,
    category,
    inventory_count,
    reorder_level,
    units_sold_last_30_days,
    ROUND(forecasted_monthly_demand, 1) as forecasted_monthly_demand,
    ROUND(days_of_inventory, 1) as days_of_inventory,
    inventory_status,
    ROUND(revenue_last_30_days, 2) as revenue_last_30_days,
    CASE 
        WHEN days_of_inventory < 15 THEN 'High Priority'
        WHEN days_of_inventory < 30 THEN 'Medium Priority'
        ELSE 'Low Priority'
    END as reorder_priority
FROM inventory_analysis
WHERE units_sold_last_90_days > 0 OR inventory_status = 'Immediate Reorder'
ORDER BY 
    CASE inventory_status 
        WHEN 'Immediate Reorder' THEN 1
        WHEN 'Monitor Closely' THEN 2
        ELSE 3
    END,
    days_of_inventory ASC;

-- =====================================
-- BUSINESS INTELLIGENCE SUMMARY VIEWS
-- =====================================

-- 8. Executive Dashboard Summary
-- Key business metrics for executive reporting
WITH date_ranges AS (
    SELECT 
        CURRENT_DATE as current_date,
        CURRENT_DATE - INTERVAL '30 days' as last_30_days,
        CURRENT_DATE - INTERVAL '60 days' as last_60_days,
        CURRENT_DATE - INTERVAL '365 days' as last_year
),
current_period AS (
    SELECT 
        COUNT(DISTINCT s.customer_key) as active_customers,
        COUNT(s.sale_key) as total_transactions,
        SUM(s.total_amount) as total_revenue,
        SUM(s.profit_amount) as total_profit,
        AVG(s.total_amount) as avg_transaction_value
    FROM fact_sales s
    JOIN dim_date d ON s.date_key = d.date_key
    WHERE d.full_date >= (SELECT last_30_days FROM date_ranges)
),
previous_period AS (
    SELECT 
        COUNT(DISTINCT s.customer_key) as prev_active_customers,
        COUNT(s.sale_key) as prev_total_transactions,
        SUM(s.total_amount) as prev_total_revenue,
        SUM(s.profit_amount) as prev_total_profit,
        AVG(s.total_amount) as prev_avg_transaction_value
    FROM fact_sales s
    JOIN dim_date d ON s.date_key = d.date_key
    WHERE d.full_date >= (SELECT last_60_days FROM date_ranges)
      AND d.full_date < (SELECT last_30_days FROM date_ranges)
)
SELECT 
    -- Current metrics
    cp.active_customers,
    cp.total_transactions,
    ROUND(cp.total_revenue, 2) as total_revenue,
    ROUND(cp.total_profit, 2) as total_profit,
    ROUND(cp.avg_transaction_value, 2) as avg_transaction_value,
    ROUND(cp.total_profit / NULLIF(cp.total_revenue, 0) * 100, 2) as profit_margin_pct,
    
    -- Growth metrics
    ROUND((cp.active_customers - pp.prev_active_customers) * 100.0 / NULLIF(pp.prev_active_customers, 0), 2) as customer_growth_pct,
    ROUND((cp.total_revenue - pp.prev_total_revenue) * 100.0 / NULLIF(pp.prev_total_revenue, 0), 2) as revenue_growth_pct,
    ROUND((cp.total_profit - pp.prev_total_profit) * 100.0 / NULLIF(pp.prev_total_profit, 0), 2) as profit_growth_pct,
    ROUND((cp.avg_transaction_value - pp.prev_avg_transaction_value) * 100.0 / NULLIF(pp.prev_avg_transaction_value, 0), 2) as avg_order_value_growth_pct
    
FROM current_period cp
CROSS JOIN previous_period pp;
