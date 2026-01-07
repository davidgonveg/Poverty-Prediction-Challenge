import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from src.data.loader import DataLoader
from src.features.preprocessor import Preprocessor
from src.models.pipeline import ModelPipeline
from src.utils.metric import calculate_poverty_metric, get_poverty_thresholds
from src.config import TARGET_COL, ID_COL, WEIGHT_COL, TRAIN_RATES

def derive_survey_id(hhid):
    return (hhid // 100000) * 100000

def validate():
    print("Starting Main Validation Loop (Leave-One-Survey-Out)...")
    
    # 1. Load Data
    loader = DataLoader()
    data = loader.load_train_data()
    
    # Derive Survey ID for Splitting
    # Assuming 'survey_id' column exists in GT, otherwise derive from hhid
    if 'survey_id' not in data.columns:
        data['survey_id'] = data[ID_COL].apply(derive_survey_id)
        
    # Get Thresholds from Rates GT file to be sure we match the competition
    rates_gt = pd.read_csv(TRAIN_RATES)
    threshold_cols, thresholds = get_poverty_thresholds(rates_gt.columns)
    print(f"Validation using {len(thresholds)} thresholds: {thresholds}")
    
    # 2. Setup Cross Validation
    logo = LeaveOneGroupOut()
    groups = data['survey_id']
    
    # Validation Results
    fold_metrics = []
    
    # Progress Bar
    # 3 Surveys -> 3 Folds
    pbar = tqdm(total=logo.get_n_splits(groups=groups), desc="CV Folds")
    
    for fold_idx, (train_idx, val_idx) in enumerate(logo.split(data, groups=groups)):
        train_df = data.iloc[train_idx]
        val_df = data.iloc[val_idx]
        
        survey_in_val = val_df['survey_id'].iloc[0]
        # tqdm.write(f"Fold {fold_idx+1}: Validating on Survey {survey_in_val}")
        
        # Split X, y
        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]
        
        X_val = val_df.drop(columns=[TARGET_COL])
        y_val = val_df[TARGET_COL]
        
        # Preprocess
        # Note: Fit on Train, Transform on Val
        preprocessor = Preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        # Train
        pipeline = ModelPipeline()
        pipeline.model.n_estimators = 100 # Reduced for valid speed, increase for final
        pipeline.train(X_train_proc, y_train)
        
        # Predict
        preds_val = pipeline.predict(X_val_proc)
        
        # Calculate Metric
        final_score, rate_mape, cons_mape = calculate_poverty_metric(
            y_true_cons=y_val.values,
            y_pred_cons=preds_val,
            hh_ids=val_df[ID_COL].values,
            weights=val_df[WEIGHT_COL].values,
            survey_ids=val_df['survey_id'].values,
            threshold_cols=threshold_cols,
            thresholds=thresholds
        )
        
        fold_metrics.append({
            'fold': fold_idx,
            'survey': survey_in_val,
            'score': final_score,
            'rate_mape': rate_mape,
            'cons_mape': cons_mape
        })
        
        pbar.set_postfix({'Score': f"{final_score:.4f}", 'Survey': survey_in_val})
        pbar.update(1)
        
    pbar.close()
    
    # Summary
    metrics_df = pd.DataFrame(fold_metrics)
    print("\nValidation Summary:")
    print(metrics_df)
    print(f"\nAverage Score: {metrics_df['score'].mean():.4f}")
    print(f"Avg Rate MAPE: {metrics_df['rate_mape'].mean():.4f}")
    print(f"Avg Cons MAPE: {metrics_df['cons_mape'].mean():.4f}")
    
    # Track Experiment
    from src.utils.tracking import ExperimentTracker
    tracker = ExperimentTracker()
    tracker.log_experiment(
        score=metrics_df['score'].mean(),
        rate_mape=metrics_df['rate_mape'].mean(),
        cons_mape=metrics_df['cons_mape'].mean()
    )

if __name__ == "__main__":
    validate()
