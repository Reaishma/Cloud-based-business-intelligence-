-- Machine Learning Support Queries for Cloud BI Platform
-- Feature engineering and model training data preparation queries

-- =====================================
-- FEATURE ENGINEERING QUERIES
-- =====================================

-- 1. Sales Forecasting Features
-- Time series features for sales prediction models
WITH daily_sales AS (
    SELECT 
        d.full_date,
        d.year,
        d.month,
        d.day_of_week,
        d.day_of_month,
        d.is_weekend,
        d.is_holiday,
        SUM(s.total_amount) as daily_revenue,
        COUNT(s.sale_key) as daily_transactions,
        AVG(s.total_amount) as avg_transaction_value,
        COUNT(DISTINCT s.customer_key) as unique_customers
    FROM fact_sales s
    JOIN dim_date d ON s.date_key = d.date_key
    GROUP BY d.full_date, d.year, d.month, d.day_of_week, d.day_of_month, d.is_weekend, d.is_holiday
),
sales_features AS (
    SELECT 
        *,
        -- Lag features
        LAG(daily_revenue, 1) OVER (ORDER BY full_date) as revenue_lag_1,
        LAG(daily_revenue, 7) OVER (ORDER BY full_date) as revenue_lag_7,
        LAG(daily_revenue, 30) OVER (ORDER BY full_date) as revenue_lag_30,
        
        -- Rolling averages
        AVG(daily_revenue) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as revenue_ma_7,
        AVG(daily_revenue) OVER (ORDER BY full_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as revenue_ma_14,
        AVG(daily_revenue) OVER (ORDER BY full_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as revenue_ma_30,
        
        -- Rolling standard deviation
        STDDEV(daily_revenue) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as revenue_std_7,
        
        -- Growth rates
        (daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY full_date)) / 
        NULLIF(LAG(daily_revenue, 1) OVER (ORDER BY full_date), 0) as daily_growth_rate,
        
        (daily_revenue - LAG(daily_revenue, 7) OVER (ORDER BY full_date)) / 
        NULLIF(LAG(daily_revenue, 7) OVER (ORDER BY full_date), 0) as weekly_growth_rate
        
    FROM daily_sales
)
SELECT 
    full_date,
    year,
    month,
    day_of_week,
    day_of_month,
    CASE WHEN is_weekend THEN 1 ELSE 0 END as is_weekend_flag,
    CASE WHEN is_holiday THEN 1 ELSE 0 END as is_holiday_flag,
    daily_revenue,
    daily_transactions,
    avg_transaction_value,
    unique_customers,
    COALESCE(revenue_lag_1, 0) as revenue_lag_1,
    COALESCE(revenue_lag_7, 0) as revenue_lag_7,
    COALESCE(revenue_lag_30, 0) as revenue_lag_30,
    COALESCE(revenue_ma_7, daily_revenue) as revenue_ma_7,
    COALESCE(revenue_ma_14, daily_revenue) as revenue_ma_14,
    COALESCE(revenue_ma_30, daily_revenue) as revenue_ma_30,
    COALESCE(revenue_std_7, 0) as revenue_volatility,
    COALESCE(daily_growth_rate, 0) as daily_growth_rate,
    COALESCE(weekly_growth_rate, 0) as weekly_growth_rate
FROM sales_features
WHERE full_date >= '2023-01-01'
ORDER BY full_date;

-- 2. Customer Churn Prediction Features
-- Comprehensive customer behavioral features for churn modeling
WITH customer_base_features AS (
    SELECT 
        c.customer_key,
        c.customer_id,
        c.customer_segment,
        c.acquisition_channel,
        EXTRACT(DAYS FROM (CURRENT_DATE - c.registration_date)) as account_age_days,
        CASE 
            WHEN c.gender = 'Male' THEN 1 
            WHEN c.gender = 'Female' THEN 0 
            ELSE 0.5 
        END as gender_encoded,
        CASE c.acquisition_channel
            WHEN 'Organic Search' THEN 1
            WHEN 'Social Media' THEN 2
            WHEN 'Google Ads' THEN 3
            WHEN 'Referral' THEN 4
            WHEN 'Email Campaign' THEN 5
            ELSE 0
        END as channel_encoded
    FROM dim_customers c
),
transaction_features AS (
    SELECT 
        s.customer_key,
        COUNT(s.sale_key) as total_transactions,
        SUM(s.total_amount) as total_spent,
        AVG(s.total_amount) as avg_order_value,
        MIN(d.full_date) as first_purchase_date,
        MAX(d.full_date) as last_purchase_date,
        EXTRACT(DAYS FROM (CURRENT_DATE - MAX(d.full_date))) as days_since_last_purchase,
        
        -- Recent activity indicators
        COUNT(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as transactions_last_30d,
        COUNT(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '60 days' THEN 1 END) as transactions_last_60d,
        COUNT(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as transactions_last_90d,
        
        SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN s.total_amount ELSE 0 END) as revenue_last_30d,
        SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '60 days' THEN s.total_amount ELSE 0 END) as revenue_last_60d,
        SUM(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '90 days' THEN s.total_amount ELSE 0 END) as revenue_last_90d,
        
        -- Purchase patterns
        STDDEV(s.total_amount) as order_value_variance,
        COUNT(DISTINCT p.category) as product_categories_purchased,
        COUNT(DISTINCT g.region) as regions_purchased_from
        
    FROM fact_sales s
    JOIN dim_date d ON s.date_key = d.date_key
    JOIN dim_products p ON s.product_key = p.product_key
    JOIN dim_geography g ON s.geography_key = g.geography_key
    GROUP BY s.customer_key
),
behavioral_features AS (
    SELECT 
        cb.customer_key,
        AVG(cb.page_views) as avg_page_views,
        AVG(cb.session_duration_minutes) as avg_session_duration,
        AVG(cb.bounce_rate) as avg_bounce_rate,
        SUM(cb.emails_opened) as total_emails_opened,
        SUM(cb.emails_sent) as total_emails_sent,
        CASE 
            WHEN SUM(cb.emails_sent) > 0 
            THEN SUM(cb.emails_opened) * 1.0 / SUM(cb.emails_sent) 
            ELSE 0 
        END as email_open_rate,
        SUM(cb.support_tickets_created) as total_support_tickets,
        SUM(cb.cart_abandonment_count) as total_cart_abandonments
    FROM fact_customer_behavior cb
    GROUP BY cb.customer_key
),
churn_labels AS (
    SELECT 
        customer_key,
        CASE 
            WHEN days_since_last_purchase > 90 THEN 1 
            ELSE 0 
        END as is_churned
    FROM transaction_features
)
SELECT 
    bf.customer_key,
    bf.customer_id,
    
    -- Demographic features
    bf.account_age_days,
    bf.gender_encoded,
    bf.channel_encoded,
    CASE bf.customer_segment
        WHEN 'High Value' THEN 2
        WHEN 'Medium Value' THEN 1
        ELSE 0
    END as segment_encoded,
    
    -- Transaction features
    COALESCE(tf.total_transactions, 0) as total_transactions,
    COALESCE(tf.total_spent, 0) as total_spent,
    COALESCE(tf.avg_order_value, 0) as avg_order_value,
    COALESCE(tf.days_since_last_purchase, 999) as days_since_last_purchase,
    COALESCE(tf.transactions_last_30d, 0) as transactions_last_30d,
    COALESCE(tf.transactions_last_60d, 0) as transactions_last_60d,
    COALESCE(tf.transactions_last_90d, 0) as transactions_last_90d,
    COALESCE(tf.revenue_last_30d, 0) as revenue_last_30d,
    COALESCE(tf.revenue_last_60d, 0) as revenue_last_60d,
    COALESCE(tf.revenue_last_90d, 0) as revenue_last_90d,
    COALESCE(tf.order_value_variance, 0) as order_value_variance,
    COALESCE(tf.product_categories_purchased, 0) as product_diversity,
    COALESCE(tf.regions_purchased_from, 0) as geographic_diversity,
    
    -- Behavioral features
    COALESCE(bhf.avg_page_views, 0) as avg_page_views,
    COALESCE(bhf.avg_session_duration, 0) as avg_session_duration,
    COALESCE(bhf.avg_bounce_rate, 0) as avg_bounce_rate,
    COALESCE(bhf.email_open_rate, 0) as email_engagement,
    COALESCE(bhf.total_support_tickets, 0) as support_interactions,
    COALESCE(bhf.total_cart_abandonments, 0) as cart_abandonment_count,
    
    -- Target variable
    COALESCE(cl.is_churned, 0) as is_churned
    
FROM customer_base_features bf
LEFT JOIN transaction_features tf ON bf.customer_key = tf.customer_key
LEFT JOIN behavioral_features bhf ON bf.customer_key = bhf.customer_key
LEFT JOIN churn_labels cl ON bf.customer_key = cl.customer_key
WHERE tf.total_transactions >= 1; -- Only customers with purchase history

-- 3. Product Recommendation Features
-- Features for collaborative filtering and content-based recommendations
WITH user_item_matrix AS (
    SELECT 
        s.customer_key,
        s.product_key,
        COUNT(s.sale_key) as purchase_frequency,
        SUM(s.total_amount) as total_spent_on_product,
        AVG(s.total_amount) as avg_spent_per_purchase,
        MAX(d.full_date) as last_purchase_date
    FROM fact_sales s
    JOIN dim_date d ON s.date_key = d.date_key
    GROUP BY s.customer_key, s.product_key
),
product_features AS (
    SELECT 
        p.product_key,
        p.category,
        p.brand,
        p.unit_price,
        p.margin_percent,
        
        -- Product popularity features
        COUNT(DISTINCT s.customer_key) as unique_customers,
        COUNT(s.sale_key) as total_sales,
        SUM(s.quantity) as total_quantity_sold,
        AVG(s.total_amount) as avg_sale_amount,
        
        -- Categorical encoding
        ROW_NUMBER() OVER (ORDER BY p.category) as category_id,
        ROW_NUMBER() OVER (ORDER BY p.brand) as brand_id
        
    FROM dim_products p
    LEFT JOIN fact_sales s ON p.product_key = s.product_key
    GROUP BY p.product_key, p.category, p.brand, p.unit_price, p.margin_percent
),
customer_preferences AS (
    SELECT 
        s.customer_key,
        
        -- Category preferences
        COUNT(DISTINCT CASE WHEN p.category = 'Electronics' THEN s.sale_key END) as electronics_purchases,
        COUNT(DISTINCT CASE WHEN p.category = 'Clothing' THEN s.sale_key END) as clothing_purchases,
        COUNT(DISTINCT CASE WHEN p.category = 'Home & Garden' THEN s.sale_key END) as home_purchases,
        COUNT(DISTINCT CASE WHEN p.category = 'Sports' THEN s.sale_key END) as sports_purchases,
        COUNT(DISTINCT CASE WHEN p.category = 'Books' THEN s.sale_key END) as books_purchases,
        
        -- Price sensitivity
        AVG(s.total_amount) as avg_purchase_amount,
        MIN(s.total_amount) as min_purchase_amount,
        MAX(s.total_amount) as max_purchase_amount,
        STDDEV(s.total_amount) as purchase_amount_variance,
        
        -- Brand loyalty
        COUNT(DISTINCT p.brand) as unique_brands_purchased,
        MODE() WITHIN GROUP (ORDER BY p.brand) as preferred_brand
        
    FROM fact_sales s
    JOIN dim_products p ON s.product_key = p.product_key
    GROUP BY s.customer_key
)
SELECT 
    uim.customer_key,
    uim.product_key,
    uim.purchase_frequency,
    uim.total_spent_on_product,
    uim.avg_spent_per_purchase,
    EXTRACT(DAYS FROM (CURRENT_DATE - uim.last_purchase_date)) as days_since_last_product_purchase,
    
    -- Product features
    pf.category_id,
    pf.brand_id,
    pf.unit_price,
    pf.margin_percent,
    pf.unique_customers as product_popularity,
    pf.total_sales as product_sales_volume,
    
    -- Customer preference features
    cp.electronics_purchases,
    cp.clothing_purchases,
    cp.home_purchases,
    cp.sports_purchases,
    cp.books_purchases,
    cp.avg_purchase_amount as customer_avg_spend,
    cp.purchase_amount_variance as customer_spend_variance,
    cp.unique_brands_purchased as customer_brand_diversity,
    
    -- Interaction features
    CASE 
        WHEN pf.unit_price <= cp.avg_purchase_amount * 0.8 THEN 1 
        ELSE 0 
    END as is_affordable,
    
    CASE 
        WHEN pf.brand_id = (
            SELECT brand_id FROM product_features pf2 
            JOIN dim_products p2 ON pf2.product_key = p2.product_key 
            WHERE p2.brand = cp.preferred_brand LIMIT 1
        ) THEN 1 ELSE 0 
    END as is_preferred_brand

FROM user_item_matrix uim
JOIN product_features pf ON uim.product_key = pf.product_key
JOIN customer_preferences cp ON uim.customer_key = cp.customer_key;

-- =====================================
-- MODEL TRAINING DATA PREPARATION
-- =====================================

-- 4. Sales Forecasting Training Data
-- Prepared dataset for time series forecasting models
CREATE OR REPLACE VIEW vw_sales_forecasting_training AS
WITH daily_aggregates AS (
    SELECT 
        d.full_date,
        SUM(s.total_amount) as daily_revenue,
        COUNT(s.sale_key) as daily_transactions,
        AVG(s.total_amount) as avg_transaction_value,
        COUNT(DISTINCT s.customer_key) as unique_customers,
        
        -- External factors (simulated)
        CASE d.day_of_week WHEN 1 THEN 0.9 WHEN 7 THEN 1.1 ELSE 1.0 END as weekend_factor,
        CASE WHEN d.is_holiday THEN 1.2 ELSE 1.0 END as holiday_factor,
        CASE d.month 
            WHEN 12 THEN 1.3 
            WHEN 11 THEN 1.1 
            WHEN 1 THEN 0.8 
            ELSE 1.0 
        END as seasonal_factor
        
    FROM fact_sales s
    JOIN dim_date d ON s.date_key = d.date_key
    WHERE d.full_date >= '2023-01-01'
    GROUP BY d.full_date, d.day_of_week, d.is_holiday, d.month
)
SELECT 
    full_date,
    daily_revenue as target,
    LAG(daily_revenue, 1) OVER (ORDER BY full_date) as lag_1_day,
    LAG(daily_revenue, 7) OVER (ORDER BY full_date) as lag_7_days,
    LAG(daily_revenue, 30) OVER (ORDER BY full_date) as lag_30_days,
    AVG(daily_revenue) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma_7_days,
    AVG(daily_revenue) OVER (ORDER BY full_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as ma_30_days,
    daily_transactions,
    avg_transaction_value,
    unique_customers,
    weekend_factor,
    holiday_factor,
    seasonal_factor,
    
    -- Trend indicators
    ROW_NUMBER() OVER (ORDER BY full_date) as time_index,
    EXTRACT(MONTH FROM full_date) as month,
    EXTRACT(DAY FROM full_date) as day_of_month,
    EXTRACT(DOW FROM full_date) as day_of_week
    
FROM daily_aggregates
ORDER BY full_date;

-- 5. Customer Lifetime Value Training Data
-- Features and target for CLV prediction
CREATE OR REPLACE VIEW vw_clv_training_data AS
WITH customer_transactions AS (
    SELECT 
        s.customer_key,
        c.customer_id,
        c.registration_date,
        COUNT(s.sale_key) as total_orders,
        SUM(s.total_amount) as total_spent,
        AVG(s.total_amount) as avg_order_value,
        MIN(d.full_date) as first_purchase_date,
        MAX(d.full_date) as last_purchase_date,
        
        -- Time-based features
        EXTRACT(DAYS FROM (MAX(d.full_date) - MIN(d.full_date))) + 1 as customer_lifespan_days,
        EXTRACT(DAYS FROM (CURRENT_DATE - c.registration_date)) as account_age_days,
        
        -- Purchase patterns
        COUNT(s.sale_key) * 1.0 / NULLIF(
            EXTRACT(DAYS FROM (MAX(d.full_date) - MIN(d.full_date))) / 30.0, 0
        ) as monthly_purchase_frequency,
        
        STDDEV(s.total_amount) as order_value_std,
        COUNT(DISTINCT p.category) as category_diversity,
        
        -- Recent activity
        COUNT(CASE WHEN d.full_date >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as orders_last_90_days
        
    FROM fact_sales s
    JOIN dim_customers c ON s.customer_key = c.customer_key
    JOIN dim_date d ON s.date_key = d.date_key
    JOIN dim_products p ON s.product_key = p.product_key
    GROUP BY s.customer_key, c.customer_id, c.registration_date
),
clv_targets AS (
    SELECT 
        customer_key,
        -- Calculate actual CLV (12-month forward looking)
        total_spent * (365.0 / NULLIF(customer_lifespan_days, 0)) as annual_clv_estimate,
        
        -- Segment customers for classification
        CASE 
            WHEN total_spent >= 2000 THEN 'High'
            WHEN total_spent >= 800 THEN 'Medium'
            ELSE 'Low'
        END as clv_segment
        
    FROM customer_transactions
)
SELECT 
    ct.customer_key,
    ct.customer_id,
    
    -- Features
    ct.account_age_days,
    ct.total_orders,
    ct.avg_order_value,
    COALESCE(ct.monthly_purchase_frequency, 0) as monthly_frequency,
    COALESCE(ct.order_value_std, 0) as order_value_std,
    ct.category_diversity,
    ct.orders_last_90_days,
    ct.customer_lifespan_days,
    
    -- Engineered features
    ct.total_spent / NULLIF(ct.total_orders, 0) as recency_weighted_aov,
    CASE WHEN ct.orders_last_90_days > 0 THEN 1 ELSE 0 END as is_active,
    
    -- Targets
    ct.total_spent as actual_clv,
    COALESCE(clv.annual_clv_estimate, ct.total_spent) as predicted_annual_clv,
    clv.clv_segment
    
FROM customer_transactions ct
LEFT JOIN clv_targets clv ON ct.customer_key = clv.customer_key
WHERE ct.total_orders >= 2 -- Focus on customers with sufficient purchase history
ORDER BY ct.total_spent DESC;
