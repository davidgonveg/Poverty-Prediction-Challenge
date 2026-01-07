import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from src.data.loader import DataLoader
from src.features.preprocessor import Preprocessor
from src.models.pipeline import ModelPipeline
from src.utils.metric import calculate_poverty_metric, get_poverty_thresholds
from src.config import TARGET_COL, ID_COL, WEIGHT_COL, TRAIN_RATES, XGB_PARAMS

def derive_survey_id(hhid):
    return (hhid // 100000) * 100000

def objective(trial):
    # 1. Suggest Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_jobs': -1,
        'random_state': 42
    }
    
    # 2. Run CV
    # Access global data (loaded once in main to save time)
    global data_global, thresholds_global, threshold_cols_global
    
    logo = LeaveOneGroupOut()
    groups = data_global['survey_id']
    
    scores = []
    rate_mapes = []
    cons_mapes = []
    
    for train_idx, val_idx in logo.split(data_global, groups=groups):
        train_df = data_global.iloc[train_idx]
        val_df = data_global.iloc[val_idx]
        
        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]
        X_val = val_df.drop(columns=[TARGET_COL])
        y_val = val_df[TARGET_COL]
        
        # Preprocess
        preprocessor = Preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        # Train
        pipeline = ModelPipeline()
        # Override default params with trial params
        pipeline.model.set_params(**params)
        pipeline.train(X_train_proc, y_train)
        
        # Predict
        preds_val = pipeline.predict(X_val_proc)
        
        # Metric
        final_score, rate_mape, cons_mape = calculate_poverty_metric(
            y_true_cons=y_val.values,
            y_pred_cons=preds_val,
            hh_ids=val_df[ID_COL].values,
            weights=val_df[WEIGHT_COL].values,
            survey_ids=val_df['survey_id'].values,
            threshold_cols=threshold_cols_global,
            thresholds=thresholds_global
        )
        scores.append(final_score)
        rate_mapes.append(rate_mape)
        cons_mapes.append(cons_mape)
        
    avg_score = np.mean(scores)
    avg_rate = np.mean(rate_mapes)
    avg_cons = np.mean(cons_mapes)
    
    # Print metrics for this trial
    print(f"Trial Metrics -> Score: {avg_score:.4f} | Rate MAPE: {avg_rate:.4f} | Cons MAPE: {avg_cons:.4f}")
    
    return avg_score

def tune():
    print("Starting Optuna Hyperparameter Tuning...")
    
    # Load Data Once
    global data_global, thresholds_global, threshold_cols_global
    loader = DataLoader()
    data_global = loader.load_train_data()
    
    if 'survey_id' not in data_global.columns:
        data_global['survey_id'] = data_global[ID_COL].apply(derive_survey_id)
        
    rates_gt = pd.read_csv(TRAIN_RATES)
    threshold_cols_global, thresholds_global = get_poverty_thresholds(rates_gt.columns)
    
    # Create Study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # Increased to 50 for serious tuning
    
    print("\n--- Tuning Completed ---")
    print("Best Score:", study.best_value)
    print("Best Params:", study.best_params)
    
    # Save Best Params
    # Ideally update src/config.py or save to a json
    import json
    with open('best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("Best params saved to best_params.json")

if __name__ == "__main__":
    tune()
