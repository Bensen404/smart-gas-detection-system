"""Central configuration for the Smart Gas AI project.

All paths are relative to the repository root so the project can be cloned and
run on any machine.  The raw CSV is treated as immutable and is protected by a
SHA-256 checksum.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
FIRMWARE_DIR = PROJECT_ROOT / "firmware" / "esp32_gas_detection"
APP_DIR = PROJECT_ROOT / "app"
DOCS_DIR = PROJECT_ROOT / "docs"

DATASET_PATH = RAW_DIR / "smart_gas_ai_enclosed_synthetic_dataset.csv"
DATASET_FILENAME = DATASET_PATH.name

# SHA-256 of the supplied dataset.  Preparation aborts if this changes.
EXPECTED_DATASET_SHA256 = (
    "82BFC7557E247D06E54C7B81AD5F2A9A294C5A44FD006274D451C6D99F1F01C7"
)

# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "session_id",
    "mq5_adc",
    "temperature_c",
    "humidity_rh_pct",
    "label",
]
FEATURE_COLUMNS_ALL = ["mq5_adc", "temperature_c", "humidity_rh_pct"]
MODEL_FEATURES = ["mq5_adc"]
TARGET_COLUMN = "label"
SESSION_COLUMN = "session_id"
CLASS_ORDER = ["Normal", "Warning", "Danger"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}

ADC_MIN = 0
ADC_MAX = 4095
HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0

# The final model deliberately uses the ADC channel only.  Grouped
# leave-one-session-out CV showed no measurable gain from temperature and
# humidity, and the deployed firmware therefore has no DHT dependency.
MODEL_INPUT_NOTE = (
    "The classifier consumes mq5_adc only. temperature_c and humidity_rh_pct "
    "are retained in the dataset for context and analysis."
)

# ---------------------------------------------------------------------------
# Reproducibility and split protocol
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SESSION_FRACTION = 3 / 18
VAL_SESSION_FRACTION = 3 / 15
CV_TIE_TOLERANCE = 0.001

# Expected assignment from the seeded GroupShuffleSplit operations.
EXPECTED_SPLIT = {
    "train": [
        "ENC_04",
        "ENC_05",
        "ENC_06",
        "ENC_07",
        "ENC_08",
        "ENC_10",
        "ENC_11",
        "ENC_12",
        "ENC_14",
        "ENC_16",
        "ENC_17",
        "ENC_18",
    ],
    "validation": ["ENC_03", "ENC_13", "ENC_15"],
    "test": ["ENC_01", "ENC_02", "ENC_09"],
}

# ---------------------------------------------------------------------------
# Candidate comparison
# ---------------------------------------------------------------------------
# Selection rule, fixed before looking at the held-out test set:
#   1. highest mean leave-one-session-out CV macro-F1 on training sessions;
#   2. if within CV_TIE_TOLERANCE, choose the smallest serialized model.
CANDIDATE_DEFINITIONS = [
    {
        "name": "dummy_most_frequent",
        "family": "DummyClassifier",
        "features": MODEL_FEATURES,
        "params": {"strategy": "most_frequent"},
        "embedded_suitability": "n/a (baseline)",
    },
    {
        "name": "logreg_adc",
        "family": "LogisticRegression",
        "features": ["mq5_adc"],
        "params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 3000,
            "class_weight": None,
            "random_state": RANDOM_SEED,
        },
        "embedded_suitability": "high",
    },
    {
        "name": "logreg_all_features",
        "family": "LogisticRegression",
        "features": FEATURE_COLUMNS_ALL,
        "params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 3000,
            "class_weight": None,
            "random_state": RANDOM_SEED,
        },
        "embedded_suitability": "high",
    },
    {
        "name": "decision_tree_depth3_adc",
        "family": "DecisionTreeClassifier",
        "features": ["mq5_adc"],
        "params": {"max_depth": 3, "random_state": RANDOM_SEED},
        "embedded_suitability": "high",
    },
    {
        "name": "random_forest_200_adc",
        "family": "RandomForestClassifier",
        "features": ["mq5_adc"],
        "params": {
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        },
        "embedded_suitability": "low",
    },
    {
        "name": "extra_trees_200_adc",
        "family": "ExtraTreesClassifier",
        "features": ["mq5_adc"],
        "params": {
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        },
        "embedded_suitability": "low",
    },
    {
        "name": "gaussian_nb_adc",
        "family": "GaussianNB",
        "features": ["mq5_adc"],
        "params": {},
        "embedded_suitability": "high",
    },
    {
        "name": "qda_adc",
        "family": "QuadraticDiscriminantAnalysis",
        "features": ["mq5_adc"],
        "params": {},
        "embedded_suitability": "medium",
    },
    {
        "name": "svc_rbf_adc",
        "family": "SVC",
        "features": ["mq5_adc"],
        "params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": RANDOM_SEED,
        },
        "embedded_suitability": "low",
    },
    {
        "name": "knn15_adc",
        "family": "KNeighborsClassifier",
        "features": ["mq5_adc"],
        "params": {"n_neighbors": 15, "weights": "uniform"},
        "embedded_suitability": "low",
    },
    {
        "name": "hist_gradient_boosting_adc",
        "family": "HistGradientBoostingClassifier",
        "features": ["mq5_adc"],
        "params": {"random_state": RANDOM_SEED},
        "embedded_suitability": "low",
    },
]

# Threshold baseline.  The two cut-points are learned on the training split
# only, never on validation or test.
BASELINE_THRESHOLD_STEP = 25

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "Normal": "#2E8B57",
    "Warning": "#D99A00",
    "Danger": "#C0392B",
}
SPLIT_COLORS = {
    "training": "#4C78A8",
    "validation": "#F58518",
    "test": "#B279A2",
}
FIGURE_DPI = 200
BOOTSTRAP_ITERATIONS = 2000
