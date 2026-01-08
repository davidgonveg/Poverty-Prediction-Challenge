import numpy as np
from scipy.optimize import minimize

class VarianceCalibrator:
    """
    Calibrates the variance of predictions to match the true distribution better.
    This is critical for poverty rate estimation which depends on the tails.
    
    Formula: y_calib = mean + alpha * (y - mean)
    """
    def __init__(self):
        self.alpha = 1.0
        self.mean = 0.0
        
    def fit(self, y_pred, y_true=None):
        # We can fit alpha to minimize Rate MAPE or just match std dev.
        # Matching std dev: alpha = std_true / std_pred
        # But since we don't have y_true on test, we assume test distribution ~ train distribution
        # Or we optimize alpha on validation set to minimize the specific metric.
        
        self.mean = np.mean(y_pred)
        
        if y_true is not None:
            # Option 1: Match Standard Deviation (Simple, Robust)
            std_pred = np.std(y_pred)
            std_true = np.std(y_true)
            if std_pred > 0:
                self.alpha = std_true / std_pred
            else:
                self.alpha = 1.0
        else:
            self.alpha = 1.0
            
    def transform(self, y_pred):
        return self.mean + self.alpha * (y_pred - self.mean)
