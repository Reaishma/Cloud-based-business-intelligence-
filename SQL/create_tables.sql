-- Data Warehouse Schema Creation for Cloud BI Platform
-- Compatible with PostgreSQL, Redshift, and BigQuery

-- =====================================
-- DIMENSION TABLES
-- =====================================

-- Customer Dimension
CREATE TABLE dim_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_key BIGINT UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(50),
    registration_date DATE,
    birth_date DATE,
    gender VARCHAR(10),
    address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    postal_code VARCHAR(20),
    customer_segment VARCHAR(50),
    lifetime_value DECIMAL(10,2),
    total_orders INTEGER,
    acquisition_channel VARCHAR(100),
    preferred_contact VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Product Dimension
CREATE TABLE dim_products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_key BIGINT UNIQUE NOT NULL,
    product_name VARCHAR(255),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    model VARCHAR(100),
    description TEXT,
    unit_cost DECIMAL(10,2),
    unit_price DECIMAL(10,2),
    margin_percent DECIMAL(5,2),
    supplier_id VARCHAR(50),
    weight_kg DECIMAL(8,3),
    dimensions_cm VARCHAR(50),
    color VARCHAR(50),
    material VARCHAR(100),
    warranty_months INTEGER,
    launch_date DATE,
    status VARCHAR(20),
    inventory_count INTEGER,
    reorder_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Date Dimension
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE UNIQUE NOT NULL,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    month_name VARCHAR(20),
    week INTEGER,
    day_of_year INTEGER,
    day_of_month INTEGER,
    day_of_week INTEGER,
    day_name VARCHAR(20),
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    fiscal_month INTEGER
);

-- Sales Representative Dimension
CREATE TABLE dim_sales_reps (
    sales_rep_id VARCHAR(50) PRIMARY KEY,
    sales_rep_key BIGINT UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    hire_date DATE,
    territory VARCHAR(100),
    region VARCHAR(100),
    manager_id VARCHAR(50),
    quota DECIMAL(12,2),
    commission_rate DECIMAL(5,4),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Geography Dimension
CREATE TABLE dim_geography (
    geography_key BIGINT PRIMARY KEY,
    region VARCHAR(100),
    country VARCHAR(100),
    state_province VARCHAR(100),
    city VARCHAR(100),
    postal_code VARCHAR(20),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    timezone VARCHAR(50),
    currency_code VARCHAR(3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================
-- FACT TABLES
-- =====================================

CURRENT_TIMESTAMP
);

-- =====================================
-- FACT TABLES
-- =====================================

-- Sales Fact Table
CREATE TABLE fact_sales (
    sale_key BIGINT PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    customer_key BIGINT REFERENCES dim_customers(customer_key),
    product_key BIGINT REFERENCES dim_products(product_key),
    sales_rep_key BIGINT REFERENCES dim_sales_reps(sales_rep_key),
    geography_key BIGINT REFERENCES dim_geography(geography_key),
    date_key INTEGER REFERENCES dim_date(date_key),
    
    -- Measures
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(12,2),
    cost_amount DECIMAL(12,2),
    profit_amount DECIMAL(12,2),
    discount_amount DECIMAL(10,2),
    tax_amount DECIMAL(10,2),
    
    -- Additional attributes
    channel VARCHAR(50),
    payment_method VARCHAR(50),
    promotion_code VARCHAR(50),
    order_priority VARCHAR(20),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customer Behavior Fact Table
CREATE TABLE fact_customer_behavior (
    behavior_key BIGINT PRIMARY KEY,
    customer_key BIGINT REFERENCES dim_customers(customer_key),
    date_key INTEGER REFERENCES dim_date(date_key),
    
    -- Website behavior
    page_views INTEGER DEFAULT 0,
    session_duration_minutes INTEGER DEFAULT 0,
    bounce_rate DECIMAL(5,4),
    
    -- Email behavior
    emails_sent INTEGER DEFAULT 0,
    emails_opened INTEGER DEFAULT 0,
    emails_clicked INTEGER DEFAULT 0,
    
    -- Purchase behavior
    cart_abandonment_count INTEGER DEFAULT 0,
    wishlist_additions INTEGER DEFAULT 0,
    product_reviews INTEGER DEFAULT 0,
    
    -- Support behavior
    support_tickets_created INTEGER DEFAULT 0,
    chat_sessions INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inventory Fact Table
CREATE TABLE fact_inventory (
    inventory_key BIGINT PRIMARY KEY,
    product_key BIGINT REFERENCES dim_products(product_key),
    geography_key BIGINT REFERENCES dim_geography(geography_key),
    date_key INTEGER REFERENCES dim_date(date_key),
    
    -- Inventory measures
    beginning_inventory INTEGER,
    ending_inventory INTEGER,
    units_received INTEGER,
    units_sold INTEGER,
    units_returned INTEGER,
    units_damaged INTEGER,
    reorder_point INTEGER,
    reorder_quantity INTEGER,
    
    -- Costs
    unit_cost DECIMAL(10,2),
    carrying_cost DECIMAL(10,2),
    stockout_cost DECIMAL(10,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================
-- ML MODEL TABLES
-- =====================================

-- Model Performance Tracking
CREATE TABLE ml_model_performance (
    model_id VARCHAR(100) PRIMARY KEY,
    model_name VARCHAR(255),
    model_type VARCHAR(100),
    version VARCHAR(50),
    
    -- Performance metrics
    accuracy DECIMAL(8,6),
    precision_score DECIMAL(8,6),
    recall_score DECIMAL(8,6),
    f1_score DECIMAL(8,6),
    auc_score DECIMAL(8,6),
    
    -- Regression metrics
    mae DECIMAL(12,6),
    rmse DECIMAL(12,6),
    r2_score DECIMAL(8,6),
    
    -- Training metadata
    training_date TIMESTAMP,
    training_samples INTEGER,
    validation_samples INTEGER,
    test_samples INTEGER,
    
    -- Model artifacts
    model_path VARCHAR(500),
    feature_importance JSON,
    hyperparameters JSON,
    
    -- Status
    status VARCHAR(50),
    is_production BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Predictions Log
CREATE TABLE ml_predictions_log (
    prediction_id BIGINT PRIMARY KEY,
    model_id VARCHAR(100) REFERENCES ml_model_performance(model_id),
    customer_key BIGINT,
    product_key BIGINT,
    
    -- Prediction details
    prediction_type VARCHAR(100),
    prediction_value DECIMAL(12,6),
    prediction_probability DECIMAL(8,6),
    confidence_score DECIMAL(8,6),
    
    -- Input features (JSON for flexibility)
    input_features JSON,
    
    -- Actual outcome (for model validation)
    actual_value DECIMAL(12,6),
    actual_outcome VARCHAR(100),
    
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    outcome_timestamp TIMESTAMP
);

-- Data Quality Monitoring
CREATE TABLE data_quality_metrics (
    metric_id BIGINT PRIMARY KEY,
    table_name VARCHAR(100),
    column_name VARCHAR(100),
    metric_type VARCHAR(50), -- completeness, accuracy, consistency, timeliness
    
    -- Metric values
    metric_value DECIMAL(8,6),
    threshold_value DECIMAL(8,6),
    is_passed BOOLEAN,
    
    -- Additional details
    row_count BIGINT,
    null_count BIGINT,
    duplicate_count BIGINT,
    
    check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================
-- INDEXES FOR PERFORMANCE
-- =====================================

-- Fact table indexes
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_key);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_key);
CREATE INDEX idx_fact_sales_date ON fact_sales(date_key);
CREATE INDEX idx_fact_sales_amount ON fact_sales(total_amount);

-- Customer behavior indexes
CREATE INDEX idx_customer_behavior_customer ON fact_customer_behavior(customer_key);
CREATE INDEX idx_customer_behavior_date ON fact_customer_behavior(date_key);

-- ML model indexes
CREATE INDEX idx_ml_predictions_model ON ml_predictions_log(model_id);
CREATE INDEX idx_ml_predictions_customer ON ml_predictions_log(customer_key);
CREATE INDEX idx_ml_predictions_timestamp ON ml_predictions_log(prediction_timestamp);

-- Data quality indexes
CREATE INDEX idx_data_quality_table ON data_quality_metrics(table_name);
CREATE INDEX idx_data_quality_timestamp ON data_quality_metrics(check_timestamp);

-- =====================================
-- VIEWS FOR ANALYTICS
-- =====================================

-- Customer 360 View
CREATE VIEW vw_customer_360 AS
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    c.customer_segment,
    c.lifetime_value,
    c.acquisition_channel,
    
    -- Sales metrics
    COUNT(s.sale_key) as total_orders,
    SUM(s.total_amount) as total_spent,
    AVG(s.total_amount) as avg_order_value,
    MAX(d.full_date) as last_purchase_date,
    
    -- Behavioral metrics
    AVG(cb.page_views) as avg_page_views,
    AVG(cb.session_duration_minutes) as avg_session_duration,
    SUM(cb.support_tickets_created) as total_support_tickets
    
FROM dim_customers c
LEFT JOIN fact_sales s ON c.customer_key = s.customer_key
LEFT JOIN dim_date d ON s.date_key = d.date_key
LEFT JOIN fact_customer_behavior cb ON c.customer_key = cb.customer_key
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, 
         c.customer_segment, c.lifetime_value, c.acquisition_channel;
-- Product Performance View
CREATE VIEW vw_product_performance AS
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    p.brand,
    p.unit_price,
    
    -- Sales metrics
    COUNT(s.sale_key) as total_orders,
    SUM(s.quantity) as total_quantity_sold,
    SUM(s.total_amount) as total_revenue,
    SUM(s.profit_amount) as total_profit,
    AVG(s.total_amount) as avg_sale_amount,
    
    -- Inventory metrics
    AVG(i.ending_inventory) as avg_inventory_level,
    SUM(i.units_sold) as total_units_moved
    
FROM dim_products p
LEFT JOIN fact_sales s ON p.product_key = s.product_key
LEFT JOIN fact_inventory i ON p.product_key = i.product_key
GROUP BY p.product_id, p.product_name, p.category, p.brand, p.unit_price;

-- Sales Performance View
CREATE VIEW vw_sales_performance AS
SELECT 
    d.year,
    d.quarter,
    d.month,
    d.month_name,
    g.region,
    g.country,
    
    -- Sales metrics
    COUNT(s.sale_key) as total_transactions,
    SUM(s.total_amount) as total_revenue,
    SUM(s.profit_amount) as total_profit,
    AVG(s.total_amount) as avg_transaction_value,
    SUM(s.quantity) as total_units_sold
    
FROM fact_sales s
JOIN dim_date d ON s.date_key = d.date_key
JOIN dim_geography g ON s.geography_key = g.geography_key
GROUP BY d.year, d.quarter, d.month, d.month_name, g.region, g.country;
