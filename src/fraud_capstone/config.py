"""Central configuration for PaySim Fraud Detection."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Auto-detect dataset path: check data/raw/paysim.csv first, then dataset/paysim.csv
_primary_data_path = DATA_DIR / "raw" / "paysim.csv"
_alt_data_path = BASE_DIR / "dataset" / "paysim.csv"
RAW_DATA_PATH = _primary_data_path if _primary_data_path.exists() else (_alt_data_path if _alt_data_path.exists() else _primary_data_path)

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_FILE = MODELS_DIR / "xgboost_fraud.joblib"
FEATURE_SCHEMA_FILE = MODELS_DIR / "feature_schema.json"
THRESHOLD_FILE = MODELS_DIR / "decision_threshold.json"

DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality.json"
METRICS_REPORT = REPORTS_DIR / "metrics.json"
CLASS_BALANCE_PLOT = REPORTS_DIR / "class_balance.png"
FRAUD_BY_TYPE_CSV = REPORTS_DIR / "fraud_by_transaction_type.csv"
CONFUSION_MATRIX_PLOT = REPORTS_DIR / "confusion_matrix.png"
DRIFT_REPORT_FILE = REPORTS_DIR / "drift_report.html"

# Dataset Columns
REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

TARGET_COLUMN = "isFraud"

PROHIBITED_FEATURE_COLUMNS = [
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
]

# Supported transaction types in PaySim
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Model hyperparameters and default threshold
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_RANDOM_STATE = 42
SPLIT_TRAIN_RATIO = 0.70
SPLIT_VAL_RATIO = 0.15
SPLIT_TEST_RATIO = 0.15
