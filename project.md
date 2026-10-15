# PaySim Fraud Detection Capstone — Project Roadmap

## 1. Project Objective

Build an end-to-end fraud detection system using the PaySim synthetic mobile-money transaction dataset.

The project covers data collection, cleaning, exploratory data analysis (EDA), feature engineering, machine learning, model evaluation, experiment tracking, API deployment, monitoring, and graph neural networks.

### Core Deliverables
- Data validation and EDA
- Leakage-safe feature engineering
- XGBoost fraud classification
- Chronological train/validation/test splits
- Model evaluation and threshold selection
- MLflow experiment tracking
- FastAPI prediction service
- Automated tests and documentation

### Advanced Extensions
- Evidently AI for drift monitoring
- Kafka for transaction streaming
- PyTorch Geometric and GraphSAGE
- Docker and Kubernetes/Minikube

**Important:** PaySim is synthetic. Results on this dataset do not establish real-world fraud detection performance.

## 2. Technology Stack

- **Language:** Python 3.11/3.12
- **Package manager:** uv
- **Data processing:** pandas, NumPy
- **Visualization:** Matplotlib
- **Machine learning:** scikit-learn, XGBoost
- **Experiment tracking:** MLflow
- **Notebooks:** Jupyter
- **API:** FastAPI, Uvicorn
- **Monitoring:** Evidently
- **Streaming:** Apache Kafka
- **Graph learning:** PyTorch, PyTorch Geometric
- **Testing:** pytest
- **Code quality:** Ruff
- **Deployment:** Docker, Kubernetes/Minikube

## 3. Project Structure

Preserve the existing `src/fraud_capstone/` structure. Integrate with existing files instead of creating duplicate modules.

```text
project1/
├── data/
│   └── raw/
│       └── paysim.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_xgboost_baseline.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_graph_neural_network.ipynb
├── models/
├── reports/
├── src/
│   └── fraud_capstone/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── eda.py
│       ├── features.py
│       ├── split.py
│       ├── evaluate.py
│       ├── train.py
│       ├── predict.py
│       ├── api.py
│       ├── drift_report.py
│       ├── kafka_producer.py
│       ├── graph_data.py
│       ├── graph_model.py
│       └── graph_train.py
├── tests/
│   ├── test_features.py
│   ├── test_split.py
│   └── test_api.py
├── pyproject.toml
├── uv.lock
├── .gitignore
├── README.md
└── PROJECT.md
```

The structure is a target layout. Keep existing equivalent files and add only what is missing.

## 4. Notebook Workflow

Use notebooks for exploration, visualization, and experiments. Move reusable code into `src/fraud_capstone/` so the final pipeline runs independently of notebooks.

- [ ] **01_data_understanding.ipynb:** Inspect dataset shape, columns, data types, missing values, duplicates, and labels.
- [ ] **02_exploratory_data_analysis.ipynb:** Analyze fraud prevalence, class imbalance, transaction types, and transaction amounts.
- [ ] **03_feature_engineering.ipynb:** Experiment with derived features and verify leakage prevention.
- [ ] **04_xgboost_baseline.ipynb:** Train baseline models and experiment with hyperparameters.
- [ ] **05_model_evaluation.ipynb:** Evaluate PR-AUC, precision, recall, F1, thresholds, and confusion matrices.
- [ ] **06_graph_neural_network.ipynb:** Explore temporal graph construction and GraphSAGE on a manageable sample.

### Notebook Guidelines
- [ ] Use chronological data splits.
- [ ] Never select the final model or threshold using test-set performance.
- [ ] Explain important decisions using Markdown cells.
- [ ] Clear unnecessary outputs before committing notebooks.
- [ ] Do not commit raw data, secrets, or large generated artifacts.
- [ ] Keep final repeatable training logic in `src/`.

## 5. Phase 1 — Environment and Data Loading

### Tasks
- [ ] Confirm Python and `uv` are installed.
- [ ] Confirm dependencies are declared in `pyproject.toml`.
- [ ] Install dependencies using `uv sync --extra dev`.
- [ ] Place the PaySim CSV in `data/raw/paysim.csv` or pass its path using `--data`.
- [ ] Validate required columns and target labels.
- [ ] Check missing values, duplicate rows, data types, and negative amounts.
- [ ] Save a quality summary to `reports/data_quality.json`.

### Acceptance Criteria
- Missing files and required columns produce clear errors.
- `isFraud` contains only valid binary labels.
- The raw dataset is excluded from Git.
- A small development sample can be loaded successfully.

## 6. Phase 2 — Exploratory Data Analysis

### Tasks
- [ ] Report the number of rows and columns.
- [ ] Calculate fraud and legitimate transaction counts and percentages.
- [ ] Quantify class imbalance and fraud prevalence.
- [ ] Compare fraud rates by transaction type.
- [ ] Analyze transaction amounts and account-balance changes.
- [ ] Save class-balance plots and summary tables.
- [ ] Record observations using actual results.

### Acceptance Criteria
- EDA can be reproduced in notebooks and Python modules.
- Figures and tables are saved under `reports/`.
- Findings and metrics are based on actual execution.

## 7. Phase 3 — Feature Engineering and Leakage Prevention

### Tasks
- [ ] Derive time features from `step`.
- [ ] Calculate amount-to-origin-balance ratios where appropriate.
- [ ] Calculate origin and destination balance deltas.
- [ ] One-hot encode transaction types.
- [ ] Exclude `isFraud` from model features.
- [ ] Exclude `isFlaggedFraud` from the baseline model.
- [ ] Exclude raw account identifiers from baseline XGBoost inputs.
- [ ] Save the training feature schema.
- [ ] Test that prohibited columns are not included in model inputs.

### Leakage Prevention Rules
- Use only information available when the transaction is scored.
- Do not use future transactions to calculate historical features.
- Fit learned preprocessing on training data only.
- Select the decision threshold using validation data, not test data.
- For graph learning, direct edges from earlier transactions to later transactions.
- Keep fraud labels separate from graph node features.

### Acceptance Criteria
- Training and inference use the same feature columns in the same order.
- Tests verify that the target and prohibited columns are excluded.
- Features do not depend on future events.

## 8. Phase 4 — XGBoost Baseline Model

### Tasks
- [ ] Sort transactions chronologically by `step`, preserving order for ties.
- [ ] Split data into 70% training, 15% validation, and 15% test.
- [ ] Train XGBoost with class weighting.
- [ ] Evaluate candidate thresholds on validation data.
- [ ] Select a threshold based on the precision/recall trade-off.
- [ ] Evaluate the finalized model on the held-out test set.
- [ ] Save the model, feature schema, threshold, and metrics.

### Required Metrics
- **PR-AUC / Average Precision:** Primary ranking metric for imbalanced fraud data.
- **Precision:** Fraction of flagged transactions that are fraudulent.
- **Recall:** Fraction of fraudulent transactions detected.
- **F1 score:** Balance between precision and recall.
- **ROC-AUC:** Ranking quality when both classes are present.
- **Confusion matrix:** True positives, false positives, true negatives, and false negatives.
- **Fraud prevalence:** Fraud rate in each data split.

### Expected Artifacts

```text
models/
├── xgboost_fraud.joblib
├── feature_schema.json
└── decision_threshold.json

reports/
├── metrics.json
├── data_quality.json
├── class_balance.png
├── fraud_by_transaction_type.csv
└── confusion_matrix.png
```

### Acceptance Criteria
- Training runs from one documented command.
- Test data is not used for model or threshold selection.
- Metrics are printed and saved.
- Actual results are recorded after execution.

## 9. Phase 5 — MLflow Experiment Tracking

### Tasks
- [ ] Install the optional MLflow dependencies.
- [ ] Start the MLflow UI.
- [ ] Log model parameters and hyperparameters.
- [ ] Log the decision threshold and evaluation metrics.
- [ ] Save the trained model as an MLflow artifact.
- [ ] Compare experiments using consistent data splits.

### Commands

```powershell
uv sync --extra tracking
uv run mlflow ui --host 127.0.0.1 --port 5000
```

In another terminal, run training with the tracking option implemented in the project:

```powershell
uv run python -m fraud_capstone.train --data ".\data\raw\paysim.csv" --tracking
```

### Acceptance Criteria
- The run appears at `http://127.0.0.1:5000`.
- MLflow metrics match the saved evaluation results.
- Local training can run without MLflow if tracking is disabled.

## 10. Phase 6 — FastAPI Prediction Service

### Tasks
- [ ] Load the trained model, feature schema, and decision threshold.
- [ ] Implement `GET /health`.
- [ ] Implement `POST /score`.
- [ ] Validate transaction input fields.
- [ ] Return fraud probability, predicted class, and applied threshold.
- [ ] Add API tests for valid inputs and missing model artifacts.
- [ ] Document sample requests in `README.md`.

### Start the API

```powershell
uv run uvicorn fraud_capstone.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the endpoints.

### Acceptance Criteria
- The API uses the same feature schema as training.
- Missing model artifacts produce a clear error response.
- The API does not fabricate predictions when the model is unavailable.

## 11. Phase 7 — Evidently Drift Monitoring

### Tasks
- [ ] Define a reference dataset and a newer dataset.
- [ ] Compare feature distributions.
- [ ] Generate and save an HTML drift report.
- [ ] Document possible causes and follow-up actions.
- [ ] Investigate drift rather than automatically retraining the model.

### Acceptance Criteria
- The drift report can be generated using a documented command or function.
- The reference and current datasets are clearly identified.
- Missing optional dependencies produce a useful error message.

## 12. Phase 8 — Kafka Streaming

### Tasks
- [ ] Start Kafka locally using the project's Compose configuration.
- [ ] Replay transaction rows to a Kafka topic.
- [ ] Build a consumer that validates events and calls shared scoring logic.
- [ ] Handle malformed events and transient failures.
- [ ] Track processed, failed, and flagged transaction counts.
- [ ] Avoid logging credentials or unnecessary sensitive data.

### Acceptance Criteria
- A small sample of events can be published and consumed.
- Malformed messages do not crash the entire consumer.
- Documentation explains that CSV replay simulates streaming rather than representing a live payment feed.

## 13. Phase 9 — Graph Neural Network

### Tasks
- [ ] Install graph dependencies using `uv`.
- [ ] Build a transaction graph using shared origin/destination accounts.
- [ ] Direct edges from past transactions to later transactions.
- [ ] Keep `isFraud` exclusively as the target label.
- [ ] Split graph nodes chronologically.
- [ ] Calculate training loss only on training nodes.
- [ ] Select model checkpoints using validation PR-AUC.
- [ ] Evaluate the model on later test nodes.
- [ ] Save graph model weights and evaluation metrics.
- [ ] Compare GraphSAGE and XGBoost on comparable evaluation periods.

### Important Constraints
- Standard PaySim is synthetic and does not provide genuine device IDs or geographic locations.
- Account-linked graph relationships may not improve model performance.
- Graphs can use substantial memory; start with a manageable sample.
- Loading the first N rows is not the same as sampling across the full timeline.
- Ensure test labels do not affect training or model selection.

### Acceptance Criteria
- No graph edge sends future transaction information into an earlier node.
- Target labels are not included in node features.
- Model weights and metrics are saved.
- Baseline and graph comparisons use comparable evaluation periods.

## 14. Phase 10 — Testing and Code Quality

### Tasks
- [ ] Run `uv run pytest`.
- [ ] Test data validation and error handling.
- [ ] Test feature leakage exclusions.
- [ ] Test chronological split ordering.
- [ ] Test feature schema alignment during inference.
- [ ] Add API tests.
- [ ] Run `uv run ruff check src tests`.
- [ ] Fix test and lint failures before submission.

### Readable Code Standards
- Use descriptive variable and function names.
- Keep functions focused on one responsibility.
- Add type hints where useful.
- Explain important decisions in comments.
- Avoid duplicated logic and unnecessary one-line expressions.
- Raise clear exceptions instead of silently hiding failures.
- Use consistent naming and formatting across modules.

### Acceptance Criteria
- Tests run from a clean environment.
- Errors explain what failed and how to fix it where practical.
- Secrets, virtual environments, raw data, and large artifacts are excluded from Git.

## 15. Phase 11 — Docker and Kubernetes

This phase is a stretch goal after the baseline pipeline and API work.

### Tasks
- [ ] Create a Dockerfile for the API.
- [ ] Make model artifacts available at runtime.
- [ ] Configure health checks.
- [ ] Test the API inside a container.
- [ ] Optionally deploy using Kubernetes/Minikube.
- [ ] Configure readiness/liveness probes and resource limits.
- [ ] Document environment variables and artifact paths.

### Acceptance Criteria
- The service starts reproducibly.
- Health checks work correctly.
- Deployment instructions are complete.

## 16. Recommended Execution Order

Complete the core project before optional extensions.

1. [ ] Environment and data loading
2. [ ] Data validation and EDA
3. [ ] Feature engineering and leakage tests
4. [ ] XGBoost training and evaluation
5. [ ] MLflow experiment tracking
6. [ ] FastAPI scoring and tests
7. [ ] Evidently drift monitoring
8. [ ] Kafka producer and consumer
9. [ ] Graph construction and GraphSAGE
10. [ ] Docker/Kubernetes deployment
11. [ ] Final documentation and results

The XGBoost pipeline and API are the core deliverables. Kafka, GraphSAGE, and Kubernetes can be completed as extensions if time or computing resources are limited.

## 17. Final Submission Checklist

- [ ] `PROJECT.md` matches the implemented project.
- [ ] `README.md` includes setup and execution instructions.
- [ ] `pyproject.toml` and `uv.lock` are present.
- [ ] The `notebooks/` folder contains completed exploratory notebooks.
- [ ] Reusable code is maintained in `src/fraud_capstone/`.
- [ ] Raw data and secrets are not committed.
- [ ] Data-quality and EDA reports are saved.
- [ ] Model, feature schema, and threshold are saved.
- [ ] Test metrics and confusion matrix are saved.
- [ ] API can score a sample transaction.
- [ ] Automated tests pass.
- [ ] MLflow, drift, Kafka, and GNN results are documented if implemented.
- [ ] Results reflect actual runs, not estimates.
- [ ] Known limitations and future improvements are documented.

## 18. Final Results Log

Complete this section after running the project.

- **Dataset path:**
- **Rows processed:**
- **Fraud prevalence:**
- **Model and version:**
- **Test PR-AUC:**
- **Test precision:**
- **Test recall:**
- **Test F1:**
- **Decision threshold:**
- **API test result:**
- **Test suite result:**
- **MLflow tracking result:**
- **Drift report result:**
- **Kafka replay result:**
- **Graph model result:**
- **Known limitations:**
- **Next improvement:**
