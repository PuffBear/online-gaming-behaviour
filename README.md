# Multi-Agent Gaming Behavior Prediction System

## Project Overview
This project implements a multi-agent AI system for predicting online gaming behavior using the Kaggle dataset. The system uses free models and implements industry-standard practices for data processing, model training, and evaluation.

## Architecture Components

### 1. Input Layer
- Data ingestion from Kaggle CSV
- API Gateway for real-time predictions

### 2. Orchestration Layer
- Task Router for workflow management
- Workflow Orchestrator using Prefect/Ray

### 3. Agent Layer
- **Preprocessing Agent**: Data cleaning and normalization
- **Feature Engineering Agent**: Feature extraction and transformation
- **Prediction Agent**: Engagement and purchase predictions
- **Reinforcement Learning Agent**: Policy learning for retention
- **Online Learning Agent**: Real-time model updates
- **Supervisor Agent**: Workflow monitoring and re-training triggers
- **Model Trainer Agent**: Training pipeline management
- **Anomaly Detection Agent**: Outlier detection
- **Evaluation Agent**: Model performance assessment
- **Data Validation Agent**: Data quality checks
- **Explainability Agent**: Model interpretability

### 4. Communication Layer
- Message Broker using Redis/RabbitMQ
- Async communication between agents

### 5. Shared Memory
- Feature Store (Feast/Redis)
- Vector DB (ChromaDB/Qdrant)
- Knowledge Graph (NetworkX)
- Model Registry (MLflow)

### 6. Monitoring & Feedback
- Human-in-the-loop review system
- Comprehensive logging (ELK/Prometheus)
- Metrics Dashboard (Grafana/Streamlit)

## Setup Instructions

### Prerequisites
```bash
python >= 3.9
pip install -r requirements.txt
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd gaming_behavior_prediction

# Install dependencies
pip install -r requirements.txt

# Download the dataset
python scripts/download_data.py

# Initialize the system
python scripts/initialize_system.py
```

### Running the System
```bash
# Start the orchestrator
python orchestrator/main.py

# Start the dashboard (in another terminal)
streamlit run dashboard/app.py

# Run evaluation
python evaluation/run_evaluation.py
```

## Project Structure
```
gaming_behavior_prediction/
├── agents/                  # Agent implementations
├── orchestrator/            # Workflow orchestration
├── communication/          # Message broker and protocols
├── shared_memory/          # Feature store, vector DB, etc.
├── monitoring/             # Logging and monitoring
├── data/                   # Data storage
├── models/                 # Trained models
├── evaluation/             # Evaluation framework
├── dashboard/              # Metrics dashboard
├── config/                 # Configuration files
├── scripts/                # Utility scripts
├── tests/                  # Test suite
└── notebooks/              # Jupyter notebooks for analysis
```

## Dataset Information
- **Source**: Kaggle - Predict Online Gaming Behavior Dataset
- **Features**: Player demographics, engagement metrics, in-game activities
- **Target**: Player behavior predictions (engagement, purchases, retention)

## Technologies Used
- **ML Framework**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch/TensorFlow (for advanced models)
- **Orchestration**: Ray/Prefect
- **Message Queue**: Redis/RabbitMQ
- **Feature Store**: Feast/Redis
- **Vector DB**: ChromaDB
- **Model Registry**: MLflow
- **Monitoring**: Prometheus/Grafana
- **Dashboard**: Streamlit

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC for classification tasks
- RMSE/MAE for regression tasks
- Model drift detection
- Hallucination detection metrics

# Multi-Agent System Execution Flow

## Main Execution Flow

```
1. scripts/initialize_system.py
   ├── config/config.py (load configurations)
   ├── config/logging_config.py (setup logging)
   └── shared_memory/feature_store.py (initialize stores)
       └── shared_memory/vector_db.py
           └── shared_memory/knowledge_graph.py

2. scripts/download_data.py
   └── data/raw/ (store raw data)

3. orchestrator/main.py (ENTRY POINT)
   ├── orchestrator/workflow_manager.py
   │   └── orchestrator/task_router.py
   │       └── orchestrator/ray_orchestrator.py
   │           └── orchestrator/scheduler.py
   │
   ├── communication/message_broker.py
   │   ├── communication/redis_client.py
   │   └── communication/event_bus.py
   │
   └── agents/supervisor_agent.py (oversees all agents)
       ├── monitoring/logger.py
       └── monitoring/metrics_collector.py
```

## Agent Pipeline Flow

```
4. agents/data_validation_agent.py
   ├── data/raw/ (read)
   ├── evaluation/validation_suite.py
   └── communication/message_protocol.py → NEXT AGENT

5. agents/preprocessing_agent.py
   ├── communication/message_protocol.py (receive)
   ├── data/raw/ (read)
   ├── data/processed/ (write)
   └── communication/message_protocol.py → NEXT AGENT

6. agents/feature_engineering_agent.py
   ├── communication/message_protocol.py (receive)
   ├── data/processed/ (read)
   ├── data/features/ (write)
   ├── shared_memory/feature_store.py (store features)
   └── communication/message_protocol.py → NEXT AGENT

7. agents/anomaly_detection_agent.py
   ├── communication/message_protocol.py (receive)
   ├── data/features/ (read)
   ├── evaluation/drift_detector.py
   └── communication/message_protocol.py → ALERT/CONTINUE
```

## Model Training Flow

```
8. agents/model_trainer_agent.py
   ├── communication/message_protocol.py (receive)
   ├── data/features/ (read)
   ├── data/splits/ (create train/test)
   ├── config/model_config.yaml (load params)
   ├── shared_memory/model_registry.py (register)
   ├── models/saved_models/ (save)
   └── communication/message_protocol.py → NEXT AGENT

9. agents/evaluation_agent.py
   ├── communication/message_protocol.py (receive)
   ├── models/saved_models/ (load)
   ├── data/splits/ (load test data)
   ├── evaluation/metrics.py
   ├── evaluation/hallucination_detector.py
   ├── evaluation/guardrails.py
   └── communication/message_protocol.py → RESULTS
```

## Prediction Flow

```
10. agents/prediction_agent.py
    ├── communication/message_protocol.py (receive)
    ├── shared_memory/model_registry.py (get best model)
    ├── shared_memory/feature_store.py (get features)
    ├── models/saved_models/ (load model)
    └── communication/message_protocol.py → PREDICTIONS

11. agents/explainability_agent.py
    ├── communication/message_protocol.py (receive predictions)
    ├── shared_memory/knowledge_graph.py (store explanations)
    └── communication/message_protocol.py → EXPLANATIONS
```

## Online Learning Flow (Parallel)

```
12. agents/online_learning_agent.py
    ├── communication/event_bus.py (subscribe to predictions)
    ├── shared_memory/cache_manager.py (buffer data)
    ├── models/checkpoints/ (incremental updates)
    └── shared_memory/model_registry.py (update model)

13. agents/reinforcement_learning_agent.py
    ├── communication/event_bus.py (subscribe to feedback)
    ├── shared_memory/vector_db.py (store experiences)
    └── models/saved_models/ (update policy)
```

## Monitoring & Feedback Flow

```
14. monitoring/prometheus_exporter.py
    ├── monitoring/metrics_collector.py (collect)
    └── monitoring/alert_manager.py
        └── monitoring/human_review.py
            └── communication/message_broker.py → TRIGGER RETRAINING

15. dashboard/app.py
    ├── dashboard/pages/overview.py
    ├── dashboard/pages/metrics.py
    ├── dashboard/pages/monitoring.py
    ├── dashboard/pages/predictions.py
    └── dashboard/components/charts.py
        └── dashboard/components/alerts.py
```

## Testing Flow

```
tests/conftest.py (setup)
├── tests/test_agents.py
├── tests/test_communication.py
├── tests/test_orchestrator.py
├── tests/test_evaluation.py
└── tests/test_integration.py (end-to-end)
```

## Deployment Flow

```
scripts/run_pipeline.py (production entry)
└── scripts/deploy.py
    └── orchestrator/main.py (starts system)
```

## Key Communication Patterns

- **Synchronous**: Agent → Message Broker → Next Agent
- **Asynchronous**: Agent → Event Bus → Multiple Subscribers
- **Shared State**: Agent → Feature Store/Vector DB → Other Agents
- **Monitoring**: All Agents → Metrics Collector → Dashboard

## Data Flow

```
Raw Data → Validation → Preprocessing → Feature Engineering → 
Feature Store → Model Training → Model Registry → 
Prediction → Explainability → Dashboard/API
```

## Feedback Loop

```
Predictions → User Feedback → Human Review → 
Supervisor Agent → Trigger Retraining → Model Trainer Agent
```

## Author
- Agriya Yadav

## License
MIT License