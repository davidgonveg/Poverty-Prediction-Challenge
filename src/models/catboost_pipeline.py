from catboost import CatBoostRegressor
import joblib
import numpy as np
import os
from src.config import RANDOM_SEED, USE_LOG_TARGET

# CatBoost Params
CAT_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MAE', # Optimize for MAE directly as we care about MAPE
    'eval_metric': 'MAE',
    'random_seed': RANDOM_SEED,
    'verbose': 0, # Manage verbosity manually
    'allow_writing_files': False
}

class CatBoostPipeline:
    def __init__(self, params=None):
        self.params = params if params else CAT_PARAMS
        self.model = CatBoostRegressor(**self.params)
        self.use_log = USE_LOG_TARGET

    def train(self, X, y, X_val=None, y_val=None):
        print(f"Training CatBoost model... (Log-Target: {self.use_log})")
        
        if self.use_log:
            y_train = np.log(y)
            if y_val is not None:
                y_val_trans = np.log(y_val)
        else:
            y_train = y
            y_val_trans = y_val if y_val is not None else None
            
        # CatBoost handles validation sets internally for early stopping if desired
        # but for consistency with other pipelines we just fit
        if X_val is not None and y_val_trans is not None:
             self.model.fit(X, y_train, eval_set=(X_val, y_val_trans), verbose=100)
        else:
             self.model.fit(X, y_train, verbose=100)

    def predict(self, X):
        preds = self.model.predict(X)
        
        if self.use_log:
            preds = np.exp(preds)
            
        return preds

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # CatBoost has its own save method, but joblib is standard here
        # self.model.save_model(path) 
        joblib.dump(self.model, path)
        print(f"CatBoost Model saved to {path}")

    def load_model(self, path):
        self.model = joblib.load(path)
        print(f"CatBoost Model loaded from {path}")
