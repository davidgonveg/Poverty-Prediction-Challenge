from xgboost import XGBRegressor
import joblib
import numpy as np
from src.config import RANDOM_SEED, USE_LOG_TARGET, XGB_PARAMS

class ModelPipeline:
    def __init__(self):
        self.model = XGBRegressor(**XGB_PARAMS)
        self.use_log = USE_LOG_TARGET

    def train(self, X, y):
        print(f"Training XGBoost model... (Log-Target: {self.use_log})")
        
        if self.use_log:
            # Shift by slight epsilon if needed, but consumption is usually > 0
            # simple log is usually preferred for econometric data
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
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
