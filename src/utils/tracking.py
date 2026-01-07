import pandas as pd
from datetime import datetime
from pathlib import Path
from src.config import DATA_DIR, XGB_PARAMS, USE_LOG_TARGET

TRACKING_FILE = DATA_DIR / "experiments.csv"

class ExperimentTracker:
    def __init__(self):
        self.path = TRACKING_FILE
        
    def log_experiment(self, score, rate_mape, cons_mape, note=""):
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': 'XGBoost',
            'use_log_target': USE_LOG_TARGET,
            'n_estimators': XGB_PARAMS['n_estimators'],
            'learning_rate': XGB_PARAMS['learning_rate'],
            'max_depth': XGB_PARAMS['max_depth'],
            'score': round(score, 5),
            'rate_mape': round(rate_mape, 5),
            'cons_mape': round(cons_mape, 5),
            'note': note
        }
        
        df_entry = pd.DataFrame([entry])
        
        if not self.path.exists():
            df_entry.to_csv(self.path, index=False)
        else:
            df_entry.to_csv(self.path, mode='a', header=False, index=False)
            
        print(f"Experiment logged to {self.path}")
