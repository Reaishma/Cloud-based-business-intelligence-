// Core Application Logic
class CloudBIDashboard {
    constructor() {
        this.currentPage = 'dashboard';
        this.currentTab = 'realtime';
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupTabs();
        this.loadInitialCharts();
        this.setupWorkflowControls();
        this.setupModelSelectors();
        this.startRealTimeUpdates();
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetPage = link.dataset.page;
                this.showPage(targetPage);
                
                // Update active nav item
                navLinks.forEach(nav => nav.classList.remove('active'));
                link.classList.add('active');
            });
        });
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;
                const container = button.closest('.tab-container, .performance-tabs, .insights-tabs');
                
                // Update tab buttons
                container.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update tab content
                container.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                container.querySelector(`#${targetTab}`).classList.add('active');
                
                // Load tab-specific content
                this.loadTabContent(targetTab);
            });
        });
    }

    showPage(pageId) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
        
        // Show target page
        document.getElementById(pageId).classList.add('active');
        
        // Load page-specific content
        this.loadPageContent(pageId);
        this.currentPage = pageId;
    }

    loadPageContent(pageId) {
        switch(pageId) {
            case 'dashboard':
                this.loadDashboardCharts();
                break;
            case 'ml-workflows':
                this.loadWorkflowCharts();
                break;
            case 'model-performance':
                this.loadPerformanceCharts();
                break;
            case 'business-insights':
                this.loadInsightsCharts();
                break;
        }
    }

    loadTabContent(tabId) {
        switch(tabId) {
            case 'realtime':
                this.loadRealtimeCharts();
                break;
            case 'predictive':
                this.loadPredictiveCharts();
                break;
            case 'overview':
                this.loadModelOverview();
                break;
        }
    }

    // Chart Loading Methods
    loadInitialCharts() {
        this.loadDashboardCharts();
    }

    loadDashboardCharts() {
        this.createSalesChart();
        this.createSegmentsChart();
        this.createRevenueChart();
    }

    createSalesChart() {
        const dates = this.generateDateRange(30);
        const salesData = dates.map(date => ({
            x: date,
            y: Math.random() * 2000 + 1000 + Math.sin(dates.indexOf(date) * 0.2) * 300
        }));

        const trace = {
            x: salesData.map(d => d.x),
            y: salesData.map(d => d.y),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Daily Sales',
            line: { color: '#FF6B6B', width: 3 },
            marker: { size: 6 }
        };

        const layout = {
            title: '',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Sales ($)' },
            margin: { t: 20, l: 60, r: 20, b: 40 },
            font: { family: 'Arial, sans-serif', size: 12 }
        };

        Plotly.newPlot('sales-chart', [trace], layout, { responsive: true });
    }

    createSegmentsChart() {
        const segments = ['High Value', 'Medium Value', 'Low Value', 'New Customers', 'At Risk'];
        const values = [350, 670, 1100, 450, 230];
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'];

        const trace = {
            labels: segments,
            values: values,
            type: 'pie',
            marker: { colors: colors },
            textinfo: 'label+percent',
            textposition: 'outside'
        };

        const layout = {
            title: '',
            margin: { t: 20, l: 20, r: 20, b: 20 },
            font: { family: 'Arial, sans-serif', size: 12 }
        };

        Plotly.newPlot('segments-chart', [trace], layout, { responsive: true });
    }

    createRevenueChart() {
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
        const revenue = [120000, 135000, 128000, 142000, 156000, 168000];
        const categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'];
        const categoryRevenue = [85000, 65000, 45000, 35000, 25000];

        const revenueTrace = {
            x: months,
            y: revenue,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Monthly Revenue',
            line: { color: '#4ECDC4', width: 3 },
            yaxis: 'y'
        };

        const categoryTrace = {
            x: categories,
            y: categoryRevenue,
            type: 'bar',
            name: 'Category Revenue',
            marker: { color: '#45B7D1' },
            yaxis: 'y2'
        };

        const layout = {
            title: '',
            xaxis: { domain: [0, 0.48] },
            xaxis2: { domain: [0.52, 1] },
            yaxis: { title: 'Revenue ($)', side: 'left' },
            yaxis2: { title: 'Category Revenue ($)', side: 'right', overlaying: 'y', anchor: 'x2' },
            margin: { t: 20, l: 60, r: 60, b: 40 },
            font: { family: 'Arial, sans-serif', size: 12 }
        };

        Plotly.newPlot('revenue-chart', [revenueTrace, categoryTrace], layout, { responsive: true });
    }

    // Predictive Charts
    loadPredictiveCharts() {
        this.createForecastChart();
        this.createCLVChart();
        this.createChurnChart();
    }

    createForecastChart() {
        const historicalDates = this.generateDateRange(30, -30);
        const forecastDates = this.generateDateRange(30, 0);
        
        const historicalSales = historicalDates.map((date, i) => 
            1000 + Math.sin(i * 0.2) * 200 + Math.random() * 100
        );
        
        const forecastSales = forecastDates.map((date, i) => 
            1100 + Math.sin(i * 0.2) * 200 + i * 5 + Math.random() * 50
        );
        
        const upperBound = forecastSales.map(val => val * 1.2);
        const lowerBound = forecastSales.map(val => val * 0.8);

        const traces = [
            {
                x: historicalDates,
                y: historicalSales,
                type: 'scatter',
                mode: 'lines',
                name: 'Historical Sales',
                line: { color: '#FF6B6B', width: 2 }
            },
            {
                x: forecastDates,
                y: forecastSales,
                type: 'scatter',
                mode: 'lines',
                name: 'Forecast',
                line: { color: '#4ECDC4', width: 2, dash: 'dash' }
            },
            {
                x: forecastDates,
                y: upperBound,
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Bound',
                line: { color: 'rgba(0,0,0,0)' },
                showlegend: false
            },
            {
                x: forecastDates,
                y: lowerBound,
                type: 'scatter',
                mode: 'lines',
                name: 'Confidence Interval',
                fill: 'tonexty',
                fillcolor: 'rgba(78, 205, 196, 0.2)',
                line: { color: 'rgba(0,0,0,0)' }
            }
        ];

        const layout = {
            title: '30-Day Sales Forecast',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Sales ($)' },
            margin: { t: 40, l: 60, r: 20, b: 40 },
            hovermode: 'x unified'
        };

        Plotly.newPlot('forecast-chart', traces, layout, { responsive: true });
    }

    createCLVChart() {
        const clvData = Array.from({length: 1000}, () => 
            Math.random() * 4000 + 200 + Math.random() * Math.random() * 2000
        );

        const trace = {
            x: clvData,
            type: 'histogram',
            nbinsx: 20,
            name: 'CLV Distribution',
            marker: { color: '#45B7D1' }
        };

        const layout = {
            title: 'Customer Lifetime Value Distribution',
            xaxis: { title: 'Predicted CLV ($)' },
            yaxis: { title: 'Number of Customers' },
            margin: { t: 40, l: 60, r: 20, b: 40 }
        };

        Plotly.newPlot('clv-chart', [trace], layout, { responsive: true });
    }

    createChurnChart() {
        const riskLevels = ['High Risk', 'Medium Risk', 'Low Risk'];
        const customerCounts = [342, 567, 891];
        const colors = ['#FF6B6B', '#FFD93D', '#6BCF7F'];

        const trace = {
            x: riskLevels,
            y: customerCounts,
            type: 'bar',
            marker: { color: colors },
            name: 'Customer Risk Distribution'
        };

        const layout = {
            title: 'Customer Churn Risk Distribution',
            xaxis: { title: 'Risk Level' },
            yaxis: { title: 'Number of Customers' },
            margin: { t: 40, l: 60, r: 20, b: 40 }
        };

        Plotly.newPlot('churn-chart', [trace], layout, { responsive: true });
    }

    // Utility Methods
    generateDateRange(days, offset = 0) {
        const dates = [];
        const today = new Date();
        
        for (let i = 0; i < days; i++) {
            const date = new Date(today);
            date.setDate(today.getDate() + offset + i);
            dates.push(date.toISOString().split('T')[0]);
        }
        
        return dates;
    }

    generateRandomData(length, min = 0, max = 100) {
        return Array.from({length}, () => Math.random() * (max - min) + min);
    }

    // Real-time Updates
    startRealTimeUpdates() {
        setInterval(() => {
            this.updateRealTimeMetrics();
        }, 30000); // Update every 30 seconds
    }

    updateRealTimeMetrics() {
        if (this.currentPage === 'dashboard') {
            // Update KPI values with small random changes
            const kpiValues = document.querySelectorAll('.kpi-value');
            kpiValues.forEach(element => {
                if (element.textContent.includes('%')) {
                    const currentValue = parseFloat(element.textContent);
                    const newValue = currentValue + (Math.random() - 0.5) * 0.2;
                    element.textContent = newValue.toFixed(1) + '%';
                } else if (element.textContent.includes(',')) {
                    const currentValue = parseInt(element.textContent.replace(',', ''));
                    const newValue = currentValue + Math.floor((Math.random() - 0.5) * 20);
                    element.textContent = newValue.toLocaleString();
                }
            });
        }
    }
}
// Workflow Methods
setupWorkflowControls() {
    const workflowSelect = document.getElementById('workflow-select');
    if (workflowSelect) {
        workflowSelect.addEventListener('change', (e) => {
            this.showWorkflowPanel(e.target.value);
        });
    }

    const horizonSlider = document.getElementById('horizon-slider');
    if (horizonSlider) {
        horizonSlider.addEventListener('input', (e) => {
            document.getElementById('horizon-value').textContent = e.target.value;
        });
    }
}

showWorkflowPanel(workflowType) {
    // Hide all workflow panels
    document.querySelectorAll('.workflow-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // Show selected panel
    const targetPanel = document.getElementById(workflowType);
    if (targetPanel) {
        targetPanel.classList.add('active');
    }
}

loadWorkflowCharts() {
    this.createWorkflowDiagram();
}

createWorkflowDiagram() {
    // Create a simple workflow visualization
    const steps = [
        'Data Source', 'Processing', 'Feature Engineering', 
        'Model Training', 'Validation', 'Deployment'
    ];
    
    const x = steps.map((_, i) => i);
    const y = steps.map(() => 1);
    
    const trace = {
        x: x,
        y: y,
        mode: 'markers+text',
        text: steps,
        textposition: 'top center',
        marker: {
            size: 20,
            color: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        },
        type: 'scatter'
    };
    
    const layout = {
        title: 'ML Pipeline Workflow',
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        margin: { t: 60, l: 20, r: 20, b: 20 },
        height: 200
    };
    
    const workflowChart = document.getElementById('workflow-diagram');
    if (workflowChart) {
        Plotly.newPlot('workflow-diagram', [trace], layout, { responsive: true });
    }
}

// Pipeline Execution
executePipeline() {
    const logContainer = document.getElementById('execution-log');
    const resultsContainer = document.getElementById('execution-results');
    
    if (!logContainer) return;
    
    const steps = [
        'Initializing SageMaker training job...',
        'Loading data from source...',
        'Preprocessing data...',
        'Feature engineering...',
        'Splitting data into train/validation...',
        'Training model...',
        'Validating model performance...',
        'Generating predictions...',
        'Saving model artifacts...',
        'Deploying model endpoint...',
        'Pipeline completed successfully!'
    ];
    
    let currentStep = 0;
    logContainer.innerHTML = '';
    
    const executeStep = () => {
        if (currentStep < steps.length) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = `[${timestamp}] ${steps[currentStep]}`;
            logContainer.innerHTML += logEntry + '\n';
            logContainer.scrollTop = logContainer.scrollHeight;
            
            currentStep++;
            setTimeout(executeStep, 1000);
        } else {
            // Show results
            if (resultsContainer) {
                resultsContainer.style.display = 'block';
                resultsContainer.innerHTML = `
                    <h4>Execution Results</h4>
                    <p><strong>Status:</strong> Successfully Completed</p>
                    <p><strong>Model Accuracy:</strong> 94.2%</p>
                    <p><strong>Training Time:</strong> 12 minutes</p>
                    <p><strong>Endpoint URL:</strong> https://sagemaker.amazonaws.com/endpoints/bi-model</p>
                `;
            }
        }
    };
    
    executeStep();
}

// Model Performance Methods
setupModelSelectors() {
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            this.loadModelPerformance(e.target.value);
        });
    }
}

loadPerformanceCharts() {
    this.createPerformanceTrendChart();
    this.createConfusionMatrix();
    this.createROCCurve();
    this.createFeatureImportanceChart();
}

createPerformanceTrendChart() {
    const dates = this.generateDateRange(30);
    const accuracy = dates.map(() => 0.92 + (Math.random() - 0.5) * 0.04);
    
    const trace = {
        x: dates,
        y: accuracy,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Daily Accuracy',
        line: { color: '#4ECDC4', width: 3 },
        marker: { size: 6 }
    };
    
    // Add trend line
    const trendY = accuracy.map((_, i) => 0.92 + (i / dates.length) * 0.02);
    const trendTrace = {
        x: dates,
        y: trendY,
        type: 'scatter',
        mode: 'lines',
        name: 'Trend',
        line: { color: 'red', width: 2, dash: 'dash' }
    };
    
    const layout = {
        title: 'Model Accuracy Over Time',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Accuracy', tickformat: '.1%' },
        margin: { t: 40, l: 60, r: 20, b: 40 }
    };
    
    Plotly.newPlot('performance-trend-chart', [trace, trendTrace], layout, { responsive: true });
}

createConfusionMatrix() {
    const matrix = [
        [85, 15],
        [12, 88]
    ];
    
    const trace = {
        z: matrix,
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: false,
        text: matrix,
        texttemplate: '%{text}',
        textfont: { size: 16, color: 'white' }
    };
    
    const layout = {
        title: 'Confusion Matrix',
        xaxis: { title: 'Predicted', tickvals: [0, 1], ticktext: ['No Churn', 'Churn'] },
        yaxis: { title: 'Actual', tickvals: [0, 1], ticktext: ['No Churn', 'Churn'] },
        margin: { t: 40, l: 60, r: 20, b: 40 },
        height: 300
    };
    
    Plotly.newPlot('confusion-matrix', [trace], layout, { responsive: true });
}

createROCCurve() {
    // Generate sample ROC curve data
    const fpr = Array.from({length: 100}, (_, i) => i / 99);
    const tpr = fpr.map(x => Math.min(1, x + 0.3 + Math.random() * 0.1));
    const auc = 0.89;
    
    const rocTrace = {
        x: fpr,
        y: tpr,
        type: 'scatter',
        mode: 'lines',
        name: `ROC Curve (AUC = ${auc.toFixed(3)})`,
        line: { color: 'blue', width: 3 }
    };
    
    const diagonalTrace = {
        x: [0, 1],
        y: [0, 1],
        type: 'scatter',
        mode: 'lines',
        name: 'Random Classifier',
        line: { color: 'red', width: 2, dash: 'dash' }
    };
    
    const layout = {
        title: 'ROC Curve',
        xaxis: { title: 'False Positive Rate' },
        yaxis: { title: 'True Positive Rate' },
        margin: { t: 40, l: 60, r: 20, b: 40 }
    };
    
    Plotly.newPlot('roc-curve', [rocTrace, diagonalTrace], layout, { responsive: true });
}

createFeatureImportanceChart() {
    const features = [
        'Days Since Last Purchase',
        'Purchase Frequency', 
        'Total Spend',
        'Customer Support Tickets',
        'Email Engagement',
        'Average Order Value'
    ];
    const importance = [0.28, 0.22, 0.18, 0.12, 0.09, 0.11];
    
    const trace = {
        x: importance,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: importance,
            colorscale: 'viridis'
        }
    };
    
    const layout = {
        title: 'Feature Importance Ranking',
        xaxis: { title: 'Importance Score' },
        margin: { t: 40, l: 150, r: 20, b: 40 }
    };
    
    Plotly.newPlot('feature-importance', [trace], layout, { responsive: true });
}

loadModelPerformance(modelName) {
    // Update performance metrics based on selected model
    const performanceMetrics = {
        'sales-forecasting': { accuracy: 94.2, precision: 92.1, recall: 91.8, f1: 91.9 },
        'customer-segmentation': { accuracy: 89.7, precision: 87.5, recall: 88.2, f1: 87.8 },
        'churn-prediction': { accuracy: 91.5, precision: 89.3, recall: 90.1, f1: 89.7 },
        'price-optimization': { accuracy: 87.3, precision: 85.2, recall: 86.8, f1: 86.0 }
    };
    
    const metrics = performanceMetrics[modelName];
    if (metrics) {
        const metricElements = document.querySelectorAll('.performance-metric .metric-value');
        if (metricElements.length >= 4) {
            metricElements[0].textContent = metrics.accuracy.toFixed(1) + '%';
            metricElements[1].textContent = metrics.precision.toFixed(1) + '%';
            metricElements[2].textContent = metrics.recall.toFixed(1) + '%';
            metricElements[3].textContent = metrics.f1.toFixed(1) + '%';
        }
    }
}

// Business Insights Methods
loadInsightsCharts() {
    this.createRevenueForecastChart();
    this.createCustomerFunnelChart();
    this.createCLVDistributionChart();
}

createRevenueForecastChart() {
    const historicalMonths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    const forecastMonths = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const historicalRevenue = [120000, 135000, 128000, 142000, 156000, 168000];
    const forecastRevenue = [175000, 182000, 189000, 195000, 203000, 210000];
    const upperBound = forecastRevenue.map(val => val * 1.15);
    const lowerBound = forecastRevenue.map(val => val * 0.85);
    
    const traces = [
        {
            x: historicalMonths,
            y: historicalRevenue,
            type: 'scatter',
            mode: 'lines',
            name: 'Historical Revenue',
            line: { color: '#2E86AB', width: 3 }
        },
        {
            x: forecastMonths,
            y: forecastRevenue,
            type: 'scatter',
            mode: 'lines',
            name: 'Forecasted Revenue',
            line: { color: '#A23B72', width: 3, dash: 'dash' }
        },
        {
            x: forecastMonths,
            y: upperBound,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(0,0,0,0)' },
            showlegend: false
        },
        {
            x: forecastMonths,
            y: lowerBound,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(162, 59, 114, 0.2)',
            line: { color: 'rgba(0,0,0,0)' },
            name: 'Confidence Interval'
        }
    ];
    
    const layout = {
        title: '6-Month Revenue Forecast',
        xaxis: { title: 'Month' },
        yaxis: { title: 'Revenue ($)' },
        margin: { t: 40, l: 80, r: 20, b: 40 },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('revenue-forecast-chart', traces, layout, { responsive: true });
}

createCustomerFunnelChart() {
    const stages = ['Awareness', 'Interest', 'Consideration', 'Purchase', 'Retention', 'Advocacy'];
    const values = [10000, 5000, 2500, 1200, 800, 300];
    
    const trace = {
        type: 'funnel',
        y: stages,
        x: values,
        textinfo: 'value+percent initial',
        marker: {
            color: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        }
    };
    
    const layout = {
        title: 'Customer Journey Funnel',
        margin: { t: 40, l: 120, r: 20, b: 40 }
    };
    
    Plotly.newPlot('customer-funnel', [trace], layout, { responsive: true });
}

createCLVDistributionChart() {
    const clvData = Array.from({length: 1000}, () => {
        return Math.random() * 3000 + 200 + Math.exp(Math.random() * 2) * 100;
    });
    
    const trace = {
        x: clvData,
        type: 'histogram',
        nbinsx: 25,
        name: 'CLV Distribution',
        marker: { color: 'lightblue' }
    };
    
    const meanCLV = clvData.reduce((a, b) => a + b) / clvData.length;
    
    const layout = {
        title: 'Customer Lifetime Value Distribution',
        xaxis: { title: 'CLV ($)' },
        yaxis: { title: 'Number of Customers' },
        margin: { t: 40, l: 60, r: 20, b: 40 },
        shapes: [{
            type: 'line',
            x0: meanCLV,
            x1: meanCLV,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: 'red', width: 2, dash: 'dash' }
        }],
        annotations: [{
            x: meanCLV,
            y: 0.8,
            yref: 'paper',
            text: `Mean CLV: $${meanCLV.toFixed(0)}`,
            showarrow: true,
            arrowhead: 2,
            arrowcolor: 'red'
        }]
    };
    
    Plotly.newPlot('clv-distribution-chart', [trace], layout, { responsive: true });
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new CloudBIDashboard();
});

// Global functions for HTML onclick events
function executePipeline() {
    window.dashboard.executePipeline();
}

function updateModelPerformance(modelName) {
    window.dashboard.loadModelPerformance(modelName);
}
// Sample Data for Charts
const SAMPLE_DATA = {
    salesData: [
        { date: '2024-01-01', amount: 1250, transactions: 45 },
        { date: '2024-01-02', amount: 1180, transactions: 42 },
        { date: '2024-01-03', amount: 1320, transactions: 48 },
        { date: '2024-01-04', amount: 1450, transactions: 52 },
        { date: '2024-01-05', amount: 1380, transactions: 49 },
        { date: '2024-01-06', amount: 1560, transactions: 58 },
        { date: '2024-01-07', amount: 1680, transactions: 61 },
        { date: '2024-01-08', amount: 1520, transactions: 55 },
        { date: '2024-01-09', amount: 1420, transactions: 51 },
        { date: '2024-01-10', amount: 1350, transactions: 48 }
    ],
    
    customerSegments: {
        'Champions': { count: 234, percentage: 12.3, color: '#FFD700' },
        'Loyal Customers': { count: 567, percentage: 29.8, color: '#4ECDC4' },
        'Potential Loyalists': { count: 456, percentage: 24.0, color: '#45B7D1' },
        'At Risk': { count: 189, percentage: 9.9, color: '#FF6B6B' },
        'Need Attention': { count: 454, percentage: 23.9, color: '#FFEAA7' }
    },
    
    modelPerformance: {
        'sales-forecasting': {
            accuracy: 94.2,
            mae: 125.43,
            rmse: 189.67,
            r2: 0.924,
            features: ['Seasonality', 'Trend', 'Marketing Spend', 'Price', 'Competition']
        },
        'churn-prediction': {
            accuracy: 91.5,
            precision: 89.3,
            recall: 90.1,
            f1: 89.7,
            features: ['Days Since Last Purchase', 'Purchase Frequency', 'Total Spend', 'Support Tickets']
        },
        'customer-segmentation': {
            silhouette: 0.67,
            inertia: 892.3,
            clusters: 5,
            features: ['Recency', 'Frequency', 'Monetary', 'Lifetime Value']
        },
        'price-optimization': {
            accuracy: 87.3,
            mae: 2.45,
            revenue_impact: 15.6,
            features: ['Current Price', 'Competitor Prices', 'Demand Elasticity', 'Cost Structure']
        }
    },
    
    businessMetrics: {
        revenue: {
            current: 2847000,
            growth: 12.3,
            forecast: [2900000, 3100000, 3250000, 3400000, 3600000, 3800000]
        },
        customers: {
            total: 1847,
            new: 247,
            churn_rate: 4.2,
            ltv_growth: 18.7
        },
        operations: {
            models_deployed: 12,
            active_pipelines: 8,
            predictions_today: 2847,
            avg_accuracy: 92.4
        }
    }
};

// Configuration for different cloud platforms
const CLOUD_CONFIG = {
    aws: {
        region: 'us-east-1',
        sagemaker: {
            training_instance: 'ml.m5.large',
            inference_instance: 'ml.t2.medium'
        },
        s3: {
            data_bucket: 'bi-platform-data-bucket',
            model_bucket: 'bi-platform-model-artifacts'
        }
    },
    
    gcp: {
        project_id: 'bi-platform-project',
        region: 'us-central1',
        vertex_ai: {
            machine_type: 'n1-standard-4',
            accelerator: 'nvidia-tesla-t4'
        },
        bigquery: {
            dataset: 'bi_platform_dataset',
            location: 'US'
        }
    },
    
    tableau: {
        server_url: 'https://bi-platform.tableauserver.com',
        site_id: 'bi-platform-site',
        workbooks: [
            'executive_dashboard',
            'sales_analytics', 
            'customer_insights',
            'ml_monitoring'
        ]
    },
    
    powerbi: {
        tenant_id: '12345678-1234-1234-1234-123456789abc',
        workspace: 'BI Platform Analytics',
        datasets: [
            'sales_analytics_dataset',
            'customer_insights_dataset',
            'ml_monitoring_dataset'
        ]
    }
};

// ML Model Templates
const ML_TEMPLATES = {
    'sales-forecasting': {
        name: 'Sales Forecasting Pipeline',
        algorithm: 'ARIMA + ML Hybrid',
        inputs: ['historical_sales', 'seasonality', 'marketing_spend'],
        outputs: ['forecast_values', 'confidence_intervals'],
        metrics: ['MAE', 'RMSE', 'MAPE', 'R²'],
        deployment: 'real-time'
    },
    
    'customer-segmentation': {
        name: 'Customer Segmentation Pipeline',
        algorithm: 'K-Means Clustering',
        inputs: ['recency', 'frequency', 'monetary', 'demographics'],
        outputs: ['segment_labels', 'cluster_centers'],
        metrics: ['Silhouette Score', 'Inertia', 'Calinski-Harabasz'],
        deployment: 'batch'
    },
    
    'churn-prediction': {
        name: 'Churn Prediction Pipeline',
        algorithm: 'Random Forest',
        inputs: ['behavior_features', 'transaction_history', 'support_interactions'],
        outputs: ['churn_probability', 'risk_factors'],
        metrics: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        deployment: 'real-time'
    },
    
    'price-optimization': {
        name: 'Price Optimization Pipeline',
        algorithm: 'XGBoost',
        inputs: ['current_price', 'competitor_prices', 'demand_elasticity'],
        outputs: ['optimal_price', 'revenue_impact'],
        metrics: ['MAE', 'RMSE', 'Business Impact'],
        deployment: 'batch'
    }
};

// UI Helper Functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value / 100);
}

function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

function generateInsightText(type, data) {
    const insights = {
        revenue: [
            `Revenue growth of ${data.growth}% indicates strong market performance`,
            `Seasonal trends show ${data.peak_month} as the highest performing month`,
            `Customer acquisition cost has improved by ${data.cac_improvement}%`
        ],
        churn: [
            `${data.high_risk} customers identified as high churn risk`,
            `Primary churn factors: ${data.top_factors.join(', ')}`,
            `Retention strategies show ${data.retention_success}% success rate`
        ],
        segmentation: [
            `${data.champions} champion customers drive ${data.champion_revenue}% of revenue`,
            `Potential loyalists segment shows ${data.growth_potential}% growth opportunity`,
            `At-risk customers require immediate attention for retention`
        ]
    };
    
    return insights[type] || ['No insights available'];
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        SAMPLE_DATA,
        CLOUD_CONFIG,
        ML_TEMPLATES,
        formatCurrency,
        formatPercentage,
        formatNumber,
        generateInsightText
    };
}
