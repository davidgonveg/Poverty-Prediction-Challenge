import argparse
import sys
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import MODEL_PATH, SUBMISSION_DIR, DATA_DIR, TARGET_COL
from src.data.loader import DataLoader
from src.features.preprocessor import Preprocessor
from src.models.pipeline import ModelPipeline
from src.utils.submission import generate_submission
from src.validate import validate

def train():
    print("Starting training pipeline...")
    loader = DataLoader()
    data = loader.load_train_data()
    print("Columns in merged data:", data.columns.tolist())
    
    # Split X, y
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]
    
    # Preprocess
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Train
    model = ModelPipeline()
    model.train(X_processed, y)
    
    # Report Training Metrics (Sanity Check)
    print("\n--- Training Metrics (Overfitting Check) ---")
    preds_train = model.predict(X_processed)
    
    # NOTE: ModelPipeline.predict already handles the inverse transformation (exp) if USE_LOG_TARGET is True.
    # So preds_train is already in the original scale. No need to exp again.
        
    # Calculate simple MAPE on Training Data
    from src.utils.metric import mean_absolute_percentage_error
    train_mape = mean_absolute_percentage_error(data[TARGET_COL], preds_train)
    print(f"Training Consumption MAPE: {train_mape:.4f}")
    
    # Context: Last Validation
    from src.config import DATA_DIR
    experiments_path = DATA_DIR / "experiments.csv"
    if experiments_path.exists():
        import pandas as pd
        try:
            exp_df = pd.read_csv(experiments_path)
            if not exp_df.empty:
                last_run = exp_df.iloc[-1]
                print("\n--- Context: Latest Validation (Unseen Data) ---")
                print(f"Global Score:     {last_run['score']:.4f} (Target metric)")
                print(f"Rate MAPE:        {last_run['rate_mape']:.4f}")
                print(f"Consumption MAPE: {last_run['cons_mape']:.4f} (Compare with Train: {train_mape:.4f})")
        except Exception:
            pass
            
    print("--------------------------------------------\n")
    
    # Create a wrapper pipeline
    full_pipeline = SkPipeline([
        ('preprocessor', preprocessor.pipeline),
        ('model', model.model)
    ])
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(full_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def predict():
    print("Starting prediction pipeline...")
    loader = DataLoader()
    test_features = loader.load_test_data()
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print("Model not found! Run train first.")
        sys.exit(1)
        
    pipeline = joblib.load(model_path)
    
    # Predict
    # Note: Preprocessor inside pipeline handles standardization. 
    # But we ensured in validation that it drops overlap.
    # We rely on the pipeline loaded.
    
    predictions = pipeline.predict(test_features)
    
    # Apply expm1 if using log target?
    # NO. The ModelPipeline.predict() method handles the inverse transform (exp).
    # BUT here we are using the SKLearn Pipeline wrapper loaded from pickle.
    # The SKLearn Pipeline contains `('model', model.model)` which is the RAW XGBRegressor.
    # It does NOT contain my `ModelPipeline` wrapper class methods!
    # !!! CRITICAL BUG !!!
    # When I wrap `model.model` (the raw XGB object) into SkPipeline, I lose the `predict` method of `ModelPipeline` wrapper which did `np.exp`.
    # I must handle the inverse transform here if the wrapper didn't do it.
    
    # Check config
    from src.config import USE_LOG_TARGET
    import numpy as np
    
    # The pipeline returns raw predictions from XGBRegressor.
    # If we trained on log, these are log(y).
    if USE_LOG_TARGET:
        print("Applying exponential inverse transform to predictions...")
        predictions = np.exp(predictions)
    
    # Generate Submission
    submission_format_dir = DATA_DIR / "submission_format"
    generate_submission(predictions, test_features, submission_format_dir)

def train_ensemble():
    print("Starting ENSEMBLE training pipeline...")
    loader = DataLoader()
    data = loader.load_train_data()
    
    # Split X, y
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]
    
    # Preprocess
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Train Ensemble
    from src.models.ensemble import EnsembleModel
    ensemble = EnsembleModel()
    ensemble.train(X_processed, y)
    
    # Save (Preprocessor needs to be saved too!)
    # We save preprocessor as 'models/preprocessor.pkl'
    # And ensemble models side by side
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(preprocessor.pipeline, model_dir / "preprocessor_ensemble.pkl")
    ensemble.save_model(str(model_dir / "model.pkl")) # Will save _xgb and _lgbm
    print("Ensemble Training Complete.")

def predict_ensemble():
    print("Starting ENSEMBLE prediction pipeline...")
    loader = DataLoader()
    test_features = loader.load_test_data()
    
    model_dir = Path("models")
    preproc_path = model_dir / "preprocessor_ensemble.pkl"
    
    if not preproc_path.exists():
        print("Ensemble not found! Run train-ensemble first.")
        sys.exit(1)
        
    # Load Preprocessor
    preprocessor_pipe = joblib.load(preproc_path)
    X_test_proc = preprocessor_pipe.transform(test_features)
    
    # Load Ensemble
    from src.models.ensemble import EnsembleModel
    ensemble = EnsembleModel()
    ensemble.load_model(str(model_dir / "model.pkl"))
    
    # Predict (Handles Log Transform internal or external?)
    # EnsembleModel.predict returns blended raw output.
    # XGBPipeline and LGBMPipeline handle exp internally IF we used the pipelines from src.
    # Let's check src/models/ensemble.py
    # It uses ModelPipeline and LGBMPipeline. Both have predict() that handles exp if config set.
    # So ensemble.predict() returns Real Dollars.
    
    predictions = ensemble.predict(X_test_proc)
    
    # Generate Submission
    submission_format_dir = DATA_DIR / "submission_format"
    generate_submission(predictions, test_features, submission_format_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "predict", "validate", "tune", "validate-ensemble", "train-ensemble", "predict-ensemble"])
    args = parser.parse_args()
    
    if args.action == "train":
        train()
    elif args.action == "predict":
        predict()
    elif args.action == "validate":
        validate()
    elif args.action == "tune":
        from src.tune import tune
        # We could pass args.trials if we added that argument
        tune()
    elif args.action == "validate-ensemble":
        from src.validate_ensemble import validate_ensemble
        validate_ensemble()
    elif args.action == "train-ensemble":
        train_ensemble()
    elif args.action == "predict-ensemble":
        predict_ensemble()
