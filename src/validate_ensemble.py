import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from src.data.loader import DataLoader
from src.features.preprocessor import Preprocessor
from src.models.pipeline import ModelPipeline
from src.models.lgbm_pipeline import LGBMPipeline
from src.utils.metric import calculate_poverty_metric, get_poverty_thresholds
from src.config import TARGET_COL, ID_COL, WEIGHT_COL, TRAIN_RATES

def derive_survey_id(hhid):
    return (hhid // 100000) * 100000

def validate_ensemble():
    print("Starting Main Validation Loop (Ensemble XGB + LGBM)...")
    
    # 1. Load Data
    loader = DataLoader()
    data = loader.load_train_data()
    
    if 'survey_id' not in data.columns:
        data['survey_id'] = data[ID_COL].apply(derive_survey_id)

    # 2. Prepare CV
    logo = LeaveOneGroupOut()
    groups = data['survey_id']
    
    # LOAD GEOMETRIC TRUTH & THRESHOLDS
    rates_gt = pd.read_csv(TRAIN_RATES)
    threshold_cols, thresholds = get_poverty_thresholds(rates_gt.columns)
    print(f"Validation using {len(thresholds)} thresholds: {[round(t, 2) for t in thresholds]}")

    results = []
    
    # 3. Validation Loop
    for fold, (train_idx, val_idx) in enumerate(tqdm(logo.split(data, groups=groups), total=logo.get_n_splits(data, groups=groups), desc="CV Folds")):
        train_df = data.iloc[train_idx]
        val_df = data.iloc[val_idx]
        survey_id = val_df['survey_id'].iloc[0]
        
        # Split X, y
        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]
        X_val = val_df.drop(columns=[TARGET_COL])
        y_val = val_df[TARGET_COL]
        
        # Preprocess
        preprocessor = Preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        # --- MODEL 1: XGBoost (The Veteran) ---
        xgb = ModelPipeline()
        xgb.train(X_train_proc, y_train)
        preds_xgb = xgb.predict(X_val_proc)
        
        # --- MODEL 2: LightGBM (The Challenger) ---
        lgbm = LGBMPipeline()
        lgbm.train(X_train_proc, y_train)
        preds_lgbm = lgbm.predict(X_val_proc)
        
        # --- BLEND (The Council) ---
        # 50/50 Split
        preds_ensemble = 0.5 * preds_xgb + 0.5 * preds_lgbm
        
        # Calculate Metric for Ensemble
        score, rate_mape, cons_mape = calculate_poverty_metric(
            y_true_cons=y_val.values,
            y_pred_cons=preds_ensemble,
            hh_ids=val_df[ID_COL].values,
            weights=val_df[WEIGHT_COL].values,
            survey_ids=val_df['survey_id'].values,
            threshold_cols=threshold_cols,
            thresholds=thresholds
        )
        
        # Log results
        results.append({
            'fold': fold,
            'survey': survey_id,
            'score': score,
            'rate_mape': rate_mape,
            'cons_mape': cons_mape
        })
        
        # Update progress bar
        tqdm.write(f"Fold {fold} Survey {survey_id}: Score={score:.4f} (Cons MAPE={cons_mape:.4f})")

    # 4. Summary
    results_df = pd.DataFrame(results)
    print("\nEnsemble Validation Summary:")
    print(results_df)
    
    avg_score = results_df['score'].mean()
    avg_rate = results_df['rate_mape'].mean()
    avg_cons = results_df['cons_mape'].mean()
    
    print(f"\nAverage Ensemble Score: {avg_score:.4f}")
    print(f"Avg Rate MAPE: {avg_rate:.4f}")
    print(f"Avg Cons MAPE: {avg_cons:.4f}")
    
    return avg_score

if __name__ == "__main__":
    validate_ensemble()
