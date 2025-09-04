# Cloud Business Intelligence Platform

A comprehensive cloud-based business intelligence platform featuring machine learning analytics, predictive insights, and enterprise-grade data visualization capabilities. This project simulates Amazon SageMaker workflows and integrates with major cloud BI platforms including Tableau, Power BI, and Google Cloud BI.

![Overview](https://github.com/Reaishma/Cloud-based-business-intelligence-/blob/main/Screenshot_20250904-122836_1.jpg)

# 🚀 Live Demo
View Live Demo https://reaishma.github.io/Cloud-based-business-intelligence-/

## 🌟 Features

### Core Analytics
- **Real-time Business Dashboards** - Executive KPIs, sales performance, and operational metrics
- **Predictive Analytics** - Sales forecasting, customer lifetime value prediction, churn analysis
- **ML Model Management** - Training pipelines, model monitoring, and deployment automation
- **Customer Intelligence** - 360-degree customer analytics with segmentation and behavioral insights

### Cloud Integration
- **AWS Integration** - SageMaker simulation, S3 data lake, Redshift warehousing
- **Google Cloud Platform** - BigQuery analytics, Vertex AI modeling, Cloud Storage
- **Enterprise BI Tools** - Tableau Server, Power BI Premium, advanced visualization

### Data Pipeline
- **ETL Processing** - Automated data extraction, transformation, and loading
- **Feature Engineering** - ML-ready feature stores and data preprocessing
- **Data Quality** - Automated monitoring, validation, and anomaly detection
- **Real-time Streaming** - Live data ingestion and processing capabilities

## 🏗️ Architecture

```

│   ├── Main Dashboard
│   ├── ML Workflows
│   ├── Model Performance
│   └── Business Insights
│
├── Data Layer
│   ├── Raw Data (CSV/JSON)
│   ├── Processed Data (Parquet)
│   ├── Feature Store
│   └── Model Artifacts
│
├── ML Pipeline
│   ├── Data Processing
│   ├── Model Training
│   ├── Model Deployment
│   └── Monitoring
│
└── Configuration
    ├── AWS Services
    ├── GCP Services
    ├── BI Platforms
    └── Security Settings
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Streamlit
- Required ML libraries (see requirements)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd cloud-bi-platform
   ```

2. **Install Dependencies**
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn
   ```

3. **Run Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

4. **Access Dashboard**
   - Open browser to `http://localhost:5000`
   - Navigate through different analytics pages using sidebar

## 📊 Dashboard Pages

### 1. Main Dashboard
- Executive KPIs and business metrics
- Real-time sales performance visualization
- Customer segment distribution
- Revenue analytics with trends

### 2. ML Workflows

![ML Workflows](https://github.com/Reaishma/Cloud-based-business-intelligence-/blob/main/Screenshot_20250904-122628_1.jpg)

- **Sales Forecasting** - ARIMA + ML hybrid models
- **Customer Segmentation** - K-means clustering with RFM analysis
- **Churn Prediction** - Random Forest classification
- **Price Optimization** - XGBoost regression models

### 3. Model Performance
- Comprehensive model evaluation metrics
- Cross-validation results and learning curves
- Feature importance and correlation analysis
- Real-time model monitoring and alerts

### 4. Business Insights
- AI-powered strategic recommendations
- Revenue intelligence and forecasting
- Customer behavior analytics
- Market opportunity identification

## 🗃️ Data Structure

### Raw Data
- `sales_data.csv` - Transaction records with customer and product details
- `customer_data.csv` - Customer profiles with demographics and preferences
- `product_data.csv` - Product catalog with pricing and inventory

### Processed Data
- Feature engineered datasets for ML training
- Customer segmentation results
- Time series forecasting data
- Model performance metrics

## ⚙️ Configuration Files

### Cloud Platform Configs
- `config/aws_config.yaml` - Amazon Web Services configuration
- `config/gcp_config.yaml` - Google Cloud Platform settings
- `config/bi_platform_config.yaml` - Main BI platform configuration

### Visualization Platform Configs
- `config/tableau_config.json` - Tableau Server integration
- `config/powerbi_config.json` - Power BI Premium configuration

## 🤖 Machine Learning Pipeline

### Data Processing (`scripts/data_processing.py`)
- ETL pipeline automation
- Feature engineering and transformation
- Data quality monitoring
- Cloud data warehouse integration

### Model Training (`scripts/model_training.py`)
- Automated ML model training
- Hyperparameter optimization
- Cross-validation and performance evaluation
- Model versioning and registry

### Model Deployment (`models/deployment_scripts.py`)
- SageMaker endpoint deployment
- Vertex AI model serving
- Docker containerization
- Performance monitoring and alerting

## 🗄️ Database Schema

### Dimensional Model
- **Fact Tables**: Sales transactions, customer behavior, inventory
- **Dimension Tables**: Customers, products, dates, geography
- **ML Tables**: Model performance, predictions log, data quality metrics

### Analytics Views
- Customer 360-degree view
- Product performance analysis
- Sales performance dashboard
- Churn risk assessment

## 📈 SQL Analytics

### Business Intelligence Queries (`sql/analytics_queries.sql`)
- Sales performance with YoY comparison
- Customer segmentation (RFM analysis)
- Product performance and inventory optimization
- Executive dashboard summary metrics


### ML Support Queries (`sql/ml_support_queries.sql`)
- Feature engineering for time series forecasting
- Customer churn prediction features
- Product recommendation engine data
- Model training data preparation

## 🧪 Testing

### Test Suite (`tests/test_models.py`)
- Unit tests for data processing pipeline
- ML model validation tests
- Data generator functionality tests
- End-to-end integration tests

Run tests:
```bash
python tests/test_models.py
```

## 🔒 Security & Compliance

- **Data Encryption**: At-rest and in-transit encryption
- **Access Control**: Role-based permissions
- **Audit Logging**: Comprehensive activity tracking
- **Data Privacy**: GDPR and CCPA compliance features

## 📋 Monitoring & Operations

### Model Monitoring
- Real-time performance tracking
- Data drift detection
- Automated retraining triggers
- Alert management system

### Infrastructure Monitoring
- System resource utilization
- API response times
- Error rate tracking
- Capacity planning metrics

## 🛠️ Development

### Project Structure
```
cloud-bi-platform/
├── app.py                      # Main Streamlit application
├── pages/                      # Dashboard pages
├── utils/                      # Core utility modules
├── scripts/                    # Data processing and ML scripts
├── sql/                        # Database queries and schema
├── config/                     # Configuration files
├── data/                       # Data storage
├── models/                     # ML model artifacts
└── tests/                      # Test suite
```

### Key Components
- **Data Generator**: Synthetic data creation for testing
- **ML Models**: Simulated machine learning workflows
- **Metrics Calculator**: Model performance evaluation
- **Visualization**: Interactive Plotly charts and dashboards

## 🌐 Cloud Deployment

### AWS Deployment
- **SageMaker**: Model training and inference
- **S3**: Data lake storage
- **Redshift**: Data warehousing
- **CloudWatch**: Monitoring and logging

### Google Cloud Deployment
- **Vertex AI**: ML model management
- **BigQuery**: Analytics warehouse
- **Cloud Storage**: Object storage
- **Cloud Monitoring**: Observability

## 📚 Documentation


### API Documentation
- RESTful API endpoints for model inference
- Authentication and authorization protocols
- Rate limiting and usage guidelines

### User Guides
- Business user dashboard navigation
- Data analyst workflow documentation
- Administrator configuration guide

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request


## 📊 Features
- **Real-time Business Dashboards** - Executive KPIs and operational metrics
- **ML Workflows** - Simulated Amazon SageMaker pipelines  
- **Model Performance** - Comprehensive ML model evaluation
- **Business Insights** - AI-powered strategic recommendations
- **Cloud Integration** - AWS, GCP, Tableau, Power BI configurations

## 🚀 Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/Reaishma/cloud-bi-platform.git
   cd cloud-bi-platform

2.  **Open locally**
      
   ```
    # Using Python
python -m http.server 8080

# Using Node.js
npx http-server . -p 8080

# Or simply open index.html in browser

```

## 🎯 Key Components

**Dashboard Pages**
**Main Dashboard**: Real-time metrics and KPIs
**ML Workflows**: Pipeline management and execution
**Model Performance**: Comprehensive model analytics
**Business Insights**: Strategic recommendations

## ✅Interactive Features

- Dynamic chart updates
- Real-time data simulation
- Responsive navigation
- Model performance monitoring
- Pipeline execution simulation

## 📈 Sample Data

**The platform includes realistic sample data for:**

- Sales transactions and forecasting
- Customer segmentation analysis
- Churn prediction metrics
- Revenue optimization insights
 
## 🔧 Customization
**Adding New Charts**

```
 function createCustomChart() {
    const trace = {
        x: ['A', 'B', 'C'],
        y: [1, 2, 3],
        type: 'bar'
    };
    
    Plotly.newPlot('chart-id', [trace]);
}
```
** Modifying Styles **
```
 .custom-component {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 12px;
    padding: 1.5rem;
}
```
 ##  🌐 Browser Support
- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers

## 📊 Performance

- Lightweight: < 500KB total size
- Fast loading: < 2 seconds initial load
- Smooth animations: 60fps interactions
- Responsive: Works on all screen sizes

## 🤝 Contributing

```
- Fork the repository
- Create feature branch (git checkout -b     feature/amazing-feature)
- Commit changes (git commit -m 'Add amazing feature')
- Push to branch (git push origin feature/amazing-feature)
- Open Pull Request

```
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  🧑‍💻 Author 

 Reaishma N 


## 🆘 Support

For support and questions:
- Create an issue in the repository
- Review configuration examples in `/config`

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with core BI functionality
- ML workflow simulation
- Cloud platform integration
- Comprehensive testing suite

---

Built with ❤️ using Streamlit, Python, and modern cloud technologies.
