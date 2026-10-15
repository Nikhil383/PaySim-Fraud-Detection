# PaySim Fraud Detection Capstone System

An end-to-end, production-grade fraud detection system built on the PaySim synthetic mobile-money transaction dataset. The project encompasses leakage-safe feature engineering, chronological validation, XGBoost classification, MLflow tracking, FastAPI serving, an interactive React frontend dashboard, Evidently drift monitoring, Kafka streaming, and PyTorch Geometric GraphSAGE.

---

## 1. Project Architecture & Directory Layout

```text
├── data/
│   └── raw/
│       └── paysim.csv            <- Standard raw dataset location (excluded from git)
├── dataset/
│   └── paysim.csv            <- Local raw PaySim dataset (excluded from git)
├── frontend/                     <- Interactive React + Vite Dashboard
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx               <- Transaction simulator & risk gauge
│       └── index.css             <- Dark glassmorphism design system
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_xgboost_baseline.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_graph_neural_network.ipynb
├── models/
│   ├── feature_schema.json       <- Ordered model feature names
│   ├── decision_threshold.json   <- Optimal probability cutoff tuned on validation set
│   └── xgboost_fraud.joblib      <- Trained serialized model artifact
├── reports/
│   ├── data_quality.json         <- Column checks, missing values, duplicates report
│   ├── class_balance.png         <- Visualization of class distribution
│   ├── fraud_by_transaction_type.csv <- Fraud counts & rates per transaction type
│   ├── confusion_matrix.png      <- Test set confusion matrix plot
│   └── metrics.json              <- PR-AUC, ROC-AUC, precision, recall, F1 results
├── src/fraud_capstone/
│   ├── __init__.py
│   ├── config.py                 <- Central configuration and path auto-detection
│   ├── data.py                   <- Loading, schema validation & data quality summary
│   ├── eda.py                    <- Imbalance metrics, group-by summaries & plots
│   ├── features.py               <- Leakage-safe feature extraction & schema alignment
│   ├── split.py                  <- Chronological (70/15/15) train/val/test splitting
│   ├── evaluate.py               <- Metrics, threshold search, confusion matrix
│   ├── train.py                  <- End-to-end training pipeline CLI
│   ├── predict.py                <- Scoring logic & predictor class
│   ├── api.py                    <- FastAPI REST prediction service with CORS
│   ├── drift_report.py           <- Evidently AI feature drift report generator
│   ├── kafka_producer.py         <- Kafka transaction streaming & consumer scoring
│   ├── graph_data.py             <- Temporal DAG construction for transactions
│   ├── graph_model.py            <- GraphSAGE PyTorch Geometric architecture
│   └── graph_train.py            <- GNN training & chronological test evaluation
├── tests/
│   ├── test_features.py          <- Prohibited column leakage & derivation tests
│   ├── test_split.py             <- Chronological ordering & partition size tests
│   └── test_api.py               <- FastAPI health, scoring, and error handling tests
├── pyproject.toml                <- Project dependencies & packaging
├── Dockerfile                    <- Container configuration for FastAPI
├── docker-compose.yml            <- Local Kafka & Zookeeper configuration
└── k8s-deployment.yaml           <- Kubernetes deployment & service manifest
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) (recommended Python package manager)
- Node.js & npm (for React frontend)

### Python Dependencies
Install packages using `uv`:

```bash
# Core dependencies + development & testing tools
uv sync --extra dev

# Optional extensions
uv sync --extra tracking     # MLflow
uv sync --extra monitoring   # Evidently AI
uv sync --extra streaming    # Kafka
uv sync --extra graph        # PyTorch & PyTorch Geometric
uv sync --extra all          # All dependencies
```

---

## 3. Usage & Execution Instructions

### A. Data Setup
The PaySim dataset is supported in either of the following locations:
- `dataset/paysim.csv` *(auto-detected in project)*
- `data/raw/paysim.csv`

The configuration automatically resolves whichever location is present. You can also specify an explicit path via the `--data` flag.

### B. Training Pipeline
Run the full training pipeline, which validates data, generates EDA reports, splits chronologically, engineers features, trains XGBoost, selects the validation threshold, evaluates the test set, and saves artifacts:

```powershell
uv run python -m fraud_capstone.train
```

Or pass an explicit path / limit rows for quick iteration:
```powershell
uv run python -m fraud_capstone.train --data "./dataset/paysim.csv" --nrows 100000
```

To enable MLflow tracking:
```powershell
uv run python -m fraud_capstone.train --data "./dataset/paysim.csv" --tracking
```

To view MLflow UI:
```powershell
uv run mlflow ui --host 127.0.0.1 --port 5000
```

### C. Running Automated Tests
Run unit and integration tests with `pytest`:
```powershell
uv run pytest
```

### D. Starting the FastAPI Prediction Service
Launch the REST API:
```powershell
uv run uvicorn fraud_capstone.api:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `GET http://127.0.0.1:8000/health`
- Real-time scoring: `POST http://127.0.0.1:8000/score`

#### Sample API Request:
```bash
curl -X POST "http://127.0.0.1:8000/score" \
     -H "Content-Type: application/json" \
     -d '{
       "step": 1,
       "type": "TRANSFER",
       "amount": 181.0,
       "nameOrig": "C1231006815",
       "oldbalanceOrg": 181.0,
       "newbalanceOrig": 0.0,
       "nameDest": "C553264065",
       "oldbalanceDest": 0.0,
       "newbalanceDest": 0.0
     }'
```

#### Sample Response:
```json
{
  "fraud_probability": 0.9421,
  "is_fraud_prediction": 1,
  "decision_threshold": 0.4200,
  "status": "success"
}
```

### E. Starting the React Frontend
To launch the interactive fraud monitoring dashboard:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. The console provides:
- **Transaction Simulator**: Test custom transactions or 1-click fraud presets (*Account Drain*, *Suspicious Cash Out*, *Standard Payment*).
- **Engine Health Monitor**: Live connection badge verifying the FastAPI ML service and model artifact state.
- **Dynamic Risk Gauge**: Displays real-time calibrated fraud risk probability, binary verdict tag, and applied decision threshold.

---

## 4. Advanced Extensions

### Evidently Drift Monitoring
Generate an HTML report comparing reference and current feature distributions:
```powershell
uv run python -m fraud_capstone.drift_report --data "./dataset/paysim.csv"
```

### Kafka Streaming Simulation
1. Start Kafka and Zookeeper:
   ```bash
   docker-compose up -d
   ```
2. Start consumer:
   ```powershell
   uv run python -m fraud_capstone.kafka_producer --mode consume
   ```
3. Replay transactions via producer:
   ```powershell
   uv run python -m fraud_capstone.kafka_producer --mode produce --limit 100
   ```

### Graph Neural Network (GraphSAGE)
Construct temporal graphs and train GraphSAGE:
```powershell
uv run python -m fraud_capstone.graph_train --data "./dataset/paysim.csv" --sample 10000
```

### Docker & Kubernetes Deployment
Build container:
```bash
docker build -t paysim-fraud-api:latest .
docker run -p 8000:8000 paysim-fraud-api:latest
```

Deploy to Kubernetes / Minikube:
```bash
kubectl apply -f k8s-deployment.yaml
```
