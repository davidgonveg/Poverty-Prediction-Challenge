from lightgbm import LGBMRegressor
import joblib
import numpy as np
import os
from src.config import RANDOM_SEED, USE_LOG_TARGET

# Default params for LightGBM (Aggressive counterpart to XGBoost)
LGBM_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31, # Different than depth, controls complexity
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1
}

class LGBMPipeline:
    def __init__(self, params=None):
        self.params = params if params else LGBM_PARAMS
        self.model = LGBMRegressor(**self.params)
        self.use_log = USE_LOG_TARGET

    def train(self, X, y):
        print(f"Training LightGBM model... (Log-Target: {self.use_log})")
        
        if self.use_log:
            y_train = np.log(y)
        else:
            y_train = y
            
        self.model.fit(X, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        
        if self.use_log:
            preds = np.exp(preds)
            
        return preds

    def save_model(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"LGBM Model saved to {path}")

    def load_model(self, path):
        self.model = joblib.load(path)
        print(f"LGBM Model loaded from {path}")
