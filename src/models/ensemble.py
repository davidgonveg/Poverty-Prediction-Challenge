from src.models.pipeline import ModelPipeline
from src.models.lgbm_pipeline import LGBMPipeline
import joblib
import os
import numpy as np

class EnsembleModel:
    def __init__(self):
        self.xgb = ModelPipeline()
        self.lgbm = LGBMPipeline()
        
    def train(self, X, y):
        # Train both models
        print("--- Ensemble Training: Model 1 (XGBoost) ---")
        self.xgb.train(X, y)
        
        print("--- Ensemble Training: Model 2 (LightGBM) ---")
        self.lgbm.train(X, y)
        
    def predict(self, X):
        # Predict with both and blend
        pred_xgb = self.xgb.predict(X)
        pred_lgbm = self.lgbm.predict(X)
        
        # 50/50 Blend
        return 0.5 * pred_xgb + 0.5 * pred_lgbm
        
    def save_model(self, path_base):
        # We need to save both models separately
        # path_base typically is "models/model.pkl"
        # We will save "models/model_xgb.pkl" and "models/model_lgbm.pkl"
        
        base_dir = os.path.dirname(path_base)
        base_name = os.path.basename(path_base).replace('.pkl', '')
        
        path_xgb = os.path.join(base_dir, f"{base_name}_xgb.pkl")
        path_lgbm = os.path.join(base_dir, f"{base_name}_lgbm.pkl")
        
        self.xgb.save_model(path_xgb)
        self.lgbm.save_model(path_lgbm)
        
        print(f"Ensemble saved to {path_xgb} and {path_lgbm}")
        
    def load_model(self, path_base):
        base_dir = os.path.dirname(path_base)
        base_name = os.path.basename(path_base).replace('.pkl', '')
        
        path_xgb = os.path.join(base_dir, f"{base_name}_xgb.pkl")
        path_lgbm = os.path.join(base_dir, f"{base_name}_lgbm.pkl")
        
        if os.path.exists(path_xgb):
            self.xgb.model = joblib.load(path_xgb)
        
        if os.path.exists(path_lgbm):
            self.lgbm.model = joblib.load(path_lgbm)
            
        print("Ensemble models loaded.")
