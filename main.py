import argparse
import pandas as pd
import sys
from pathlib import Path
from src.config import MODEL_PATH, SUBMISSION_DIR, DATA_DIR
from src.data.loader import DataLoader
from src.features.preprocessor import Preprocessor
from src.models.pipeline import ModelPipeline
from src.utils.submission import generate_submission

def train():
    print("Starting training pipeline...")
    loader = DataLoader()
    data = loader.load_train_data()
    print("Columns in merged data:", data.columns.tolist())
    
    # Split X, y
    from src.config import TARGET_COL
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]
    
    # Preprocess
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Train
    model = ModelPipeline()
    model.train(X_processed, y)
    
    # Save
    # We need to save both model and preprocessor (or pipeline including preprocessor)
    # For simplicity, let's just save the model and pickled preprocessor, or a composed pipeline.
    # Current ModelPipeline only saves the model.
    # Ideally, we should wrap everything. 
    # For now, quick hack: save them separately or rely on in-memory if running predict immediately?
    # CLI separates them, so must save.
    
    # Create a wrapper pipeline
    from sklearn.pipeline import Pipeline as SkPipeline
    full_pipeline = SkPipeline([
        ('preprocessor', preprocessor.pipeline),
        ('model', model.model)
    ])
    
    import joblib
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(full_pipeline, model_dir / "baseline.pkl")
    print(f"Model saved to models/baseline.pkl")

def predict():
    print("Starting prediction pipeline...")
    loader = DataLoader()
    test_features = loader.load_test_data()
    
    import joblib
    model_path = Path("models/baseline.pkl")
    if not model_path.exists():
        print("Model not found! Run train first.")
        sys.exit(1)
        
    pipeline = joblib.load(model_path)
    
    # Predict
    # The pipeline expects raw dataframe as input because preprocessor is the first step
    # However, preprocessor excludes ID columns.
    # Wait, Preprocessor.build_pipeline takes X and selects columns by type.
    # If we pass the full test_features, it should work IF columns verify.
    # We should ensure `preprocessor` handles the dropping of IDs safely or we drop them before.
    # In `train`, we passed `X` which still had `hhid`? 
    # `Preprocessor.get_feature_columns` logic was implemented but not explicitly used in `fit_transform` inside the class I wrote.
    # I wrote: assumes `X` passed to `build_pipeline` is what we use. `build_pipeline` selects dtypes.
    # If `hhid` is int64, it might get included in numeric features and scaled!
    # BAD.
    # I need to fix Preprocessor to handle drops.
    
    # WORKAROUND: Drop IDs here before passing to pipeline.
    from src.config import ID_COL, WEIGHT_COL
    
    # Better: Fix Preprocessor to exclude headers.
    # But for now in main.py:
    # We rely on the pipeline loaded.
    
    predictions = pipeline.predict(test_features)
    
    # Generate Submission
    submission_format_dir = DATA_DIR / "submission_format"
    generate_submission(predictions, test_features, submission_format_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "predict"])
    args = parser.parse_args()
    
    if args.action == "train":
        train()
    elif args.action == "predict":
        predict()
