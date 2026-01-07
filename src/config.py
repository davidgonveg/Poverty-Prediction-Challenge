from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_FEATURES = DATA_DIR / "train_hh_features.csv"
TRAIN_LABELS = DATA_DIR / "train_hh_gt.csv"
TRAIN_RATES = DATA_DIR / "train_rates_gt.csv"
TEST_FEATURES = DATA_DIR / "test_hh_features.csv"
SUBMISSION_DIR = DATA_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

MODEL_PATH = "models/model.pkl"

# Columns
ID_COL = "hhid"
SURVEY_COL = "strata" 
SURVEY_ID_COL = "country" 

TARGET_COL = "cons_ppp17"
WEIGHT_COL = "weight"

# Flags
USE_LOG_TARGET = True

# XGBoost Params (Regularized - To Combat Overfitting)
# Rationale: Previous tuning overfit (Train MAPE 6% vs Valid 30%).
# We force simpler trees and penalize weights.
XGB_PARAMS = {
    'n_estimators': 1500, # Increased to compensate for lower LR
    'learning_rate': 0.05, # Slower learning
    'max_depth': 5, # Reduced from 8 to prevents specific memorization
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'min_child_weight': 5, # Require more samples per leaf
    'reg_alpha': 1.0, # L1 Regularization (Feature Selection)
    'reg_lambda': 5.0, # L2 Regularization (Weight Penalty)
    'n_jobs': -1,
    'random_state': 42
}

# Random Seed
RANDOM_SEED = 42
