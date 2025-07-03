"""
Model Deployment Scripts for Cloud BI Platform
Handles model deployment to AWS SageMaker and GCP AI Platform
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any
import joblib
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployment:
    """Handle model deployment and versioning"""
    
    def __init__(self):
        self.deployment_config = self._load_deployment_config()
        self.model_registry = {}
    
    def _load_deployment_config(self) -> Dict:
        """Load deployment configuration"""
        try:
            with open('config/aws_config.yaml', 'r') as f:
                import yaml
                aws_config = yaml.safe_load(f)
            
            with open('config/gcp_config.yaml', 'r') as f:
                gcp_config = yaml.safe_load(f)
                
            return {
                'aws': aws_config,
                'gcp': gcp_config
            }
        except Exception as e:
            logger.warning(f"Could not load deployment config: {e}")
            return {}
    
    def register_model(self, model_name: str, model_path: str, model_metadata: Dict):
        """Register a trained model for deployment"""
        logger.info(f"Registering model: {model_name}")
        
        model_info = {
            'model_name': model_name,
            'model_path': model_path,
            'version': self._generate_version(),
            'metadata': model_metadata,
            'registration_time': datetime.now().isoformat(),
            'status': 'registered'
        }
        
        self.model_registry[model_name] = model_info
        
        # Save registry
        with open('data/models/model_registry.json', 'w') as f:
            json.dump(self.model_registry, f, indent=2)
        
        logger.info(f"Model {model_name} registered successfully")
        return model_info
    
    def _generate_version(self) -> str:
        """Generate model version string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def deploy_to_sagemaker(self, model_name: str) -> Dict:
        """Deploy model to AWS SageMaker (simulated)"""
        logger.info(f"Deploying {model_name} to SageMaker")
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_info = self.model_registry[model_name]
        
        # Simulate SageMaker deployment
        deployment_info = {
            'endpoint_name': f"bi-platform-{model_name.replace('_', '-')}",
            'endpoint_config_name': f"bi-platform-{model_name.replace('_', '-')}-config",
            'model_name': f"bi-platform-{model_name.replace('_', '-')}-model",
            'instance_type': 'ml.t2.medium',
            'initial_instance_count': 1,
            'deployment_time': datetime.now().isoformat(),
            'status': 'InService',
            'endpoint_url': f"https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/bi-platform-{model_name.replace('_', '-')}/invocations"
        }
        
        # Update model registry
        model_info['sagemaker_deployment'] = deployment_info
        model_info['status'] = 'deployed_sagemaker'
        
        logger.info(f"Model {model_name} deployed to SageMaker endpoint: {deployment_info['endpoint_name']}")
        return deployment_info
    
    def deploy_to_vertex_ai(self, model_name: str) -> Dict:
        """Deploy model to Google Cloud Vertex AI (simulated)"""
        logger.info(f"Deploying {model_name} to Vertex AI")
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_info = self.model_registry[model_name]
        
        # Simulate Vertex AI deployment
        deployment_info = {
            'endpoint_name': f"projects/bi-platform-project/locations/us-central1/endpoints/{model_name.replace('_', '-')}-endpoint",
            'model_name': f"projects/bi-platform-project/locations/us-central1/models/{model_name.replace('_', '-')}-model",
            'deployed_model_id': f"{model_name.replace('_', '-')}-deployed-{self._generate_version()}",
            'machine_type': 'n1-standard-2',
            'min_replica_count': 1,
            'max_replica_count': 3,
            'deployment_time': datetime.now().isoformat(),
            'status': 'deployed',
            'prediction_url': f"https://us-central1-aiplatform.googleapis.com/v1/projects/bi-platform-project/locations/us-central1/endpoints/{model_name.replace('_', '-')}-endpoint:predict"
        }
        
        # Update model registry
        model_info['vertex_ai_deployment'] = deployment_info
        model_info['status'] = 'deployed_vertex_ai'
        
        logger.info(f"Model {model_name} deployed to Vertex AI endpoint")
        return deployment_info
    
    def create_inference_script(self, model_name: str) -> str:
        """Create inference script for model serving"""
        script_content = f'''
import joblib
import json
import numpy as np
from typing import Dict, Any, List

class {model_name.title().replace('_', '')}Predictor:
    """Inference script for {model_name} model"""
    
    def __init__(self, model_path: str):
        self.model_artifacts = joblib.load(model_path)
        self.model = self.model_artifacts['model']
        self.scaler = self.model_artifacts.get('scaler')
        self.feature_names = self.model_artifacts.get('feature_names', [])
      def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions on input data"""
        try:
            # Extract features
            features = []
            for feature_name in self.feature_names:
                features.append(input_data.get(feature_name, 0))
            
            # Convert to numpy array
            X = np.array([features])
            
            # Scale features if scaler is available
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict(X)[0]
                probabilities = self.model.predict_proba(X)[0].tolist()
                
                return {{
                    'prediction': int(prediction),
                    'probabilities': probabilities,
                    'model_name': '{model_name}',
                    'timestamp': str(np.datetime64('now'))
                }}
            else:
                prediction = self.model.predict(X)[0]
                return {{
                    'prediction': float(prediction),
                    'model_name': '{model_name}',
                    'timestamp': str(np.datetime64('now'))
                }}
                
        except Exception as e:
            return {{
                'error': str(e),
                'model_name': '{model_name}',
                'timestamp': str(np.datetime64('now'))
            }}

def model_fn(model_dir: str):
    """Load model for SageMaker"""
    return {model_name.title().replace('_', '')}Predictor(
        f"{{model_dir}}/{model_name}_model.pkl"
    )

def predict_fn(input_data: Dict, model):
    """Make prediction for SageMaker"""
    return model.predict(input_data)

def input_fn(request_body: str, content_type: str = 'application/json'):
    """Parse input for SageMaker"""
    if content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {{content_type}}")

def output_fn(prediction: Dict, accept: str = 'application/json'):
    """Format output for SageMaker"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {{accept}}")
'''
        
        # Save inference script
        script_path = f'models/{model_name}_inference.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Inference script created: {script_path}")
        return script_path
    
    def create_docker_config(self, model_name: str) -> str:
        """Create Docker configuration for model deployment"""
        dockerfile_content = f'''
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY {model_name}_model.pkl ./model/
COPY {model_name}_inference.py ./code/

# Set environment variables
ENV PYTHONPATH="/opt/ml/code:$PYTHONPATH"
ENV MODEL_NAME="{model_name}"

# Expose port
EXPOSE 8080

# Run inference server
CMD ["python", "code/{model_name}_inference.py"]
'''
        
        dockerfile_path = f'models/Dockerfile.{model_name}'
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt
        requirements_content = '''
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
flask>=2.0.0
gunicorn>=20.0.0
'''
        
        with open(f'models/requirements_{model_name}.txt', 'w') as f:
            f.write(requirements_content)
        
        logger.info(f"Docker configuration created: {dockerfile_path}")
        return dockerfile_path
    
    def monitor_deployment(self, model_name: str) -> Dict:
        """Monitor deployed model health and performance"""
        logger.info(f"Monitoring deployment for {model_name}")
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_info = self.model_registry[model_name]
        
        # Simulate monitoring metrics
        monitoring_data = {
            'model_name': model_name,
            'status': 'healthy',
            'metrics': {
                'response_time_ms': np.random.uniform(50, 200),
                'requests_per_minute': np.random.randint(10, 100),
                'error_rate': np.random.uniform(0, 0.05),
                'cpu_utilization': np.random.uniform(30, 80),
                'memory_utilization': np.random.uniform(40, 85)
            },
            'data_drift': {
                'drift_score': np.random.uniform(0, 0.3),
                'threshold': 0.5,
                'status': 'normal'
            },
            'model_performance': {
                'accuracy': np.random.uniform(0.85, 0.95),
                'precision': np.random.uniform(0.82, 0.92),
                'recall': np.random.uniform(0.80, 0.90)
            },
            'last_check': datetime.now().isoformat()
        }
        
        # Save monitoring data
        monitoring_path = f'data/models/{model_name}_monitoring.json'
        with open(monitoring_path, 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        
        logger.info(f"Monitoring data saved: {monitoring_path}")
        return monitoring_data
    
    def rollback_deployment(self, model_name: str, target_version: str) -> Dict:
        """Rollback model deployment to previous version"""
        logger.info(f"Rolling back {model_name} to version {target_version}")
        
        rollback_info = {
            'model_name': model_name,
            'target_version': target_version,
            'rollback_time': datetime.now().isoformat(),
            'status': 'completed',
            'reason': 'Performance degradation detected'
        }
        
        # Update model registry
        if model_name in self.model_registry:
            self.model_registry[model_name]['rollback_history'] = rollback_info
        
        logger.info(f"Model {model_name} rolled back successfully")
        return rollback_info

def main():
    """Example usage of deployment scripts"""
    deployer = ModelDeployment()
    
    # Example: Deploy churn prediction model
    try:
        # Register model
        model_metadata = {
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'performance': {'accuracy': 0.891, 'f1_score': 0.835}
        }
        
        deployer.register_model(
            'churn_prediction',
            'data/models/churn_prediction_model.pkl',
            model_metadata
        )
        
        # Create deployment artifacts
        deployer.create_inference_script('churn_prediction')
        deployer.create_docker_config('churn_prediction')
        
        # Deploy to cloud platforms
        sagemaker_deployment = deployer.deploy_to_sagemaker('churn_prediction')
        vertex_deployment = deployer.deploy_to_vertex_ai('churn_prediction')
        
        # Monitor deployment
        monitoring_data = deployer.monitor_deployment('churn_prediction')
        
        print("✅ Model deployment completed successfully")
        print(f"SageMaker endpoint: {sagemaker_deployment['endpoint_name']}")
        print(f"Vertex AI endpoint: {vertex_deployment['endpoint_name']}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")

if __name__ == "__main__":
    main()
    
