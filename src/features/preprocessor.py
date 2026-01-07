import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.config import ID_COL, TARGET_COL, WEIGHT_COL

class Preprocessor:
    def __init__(self):
        self.pipeline = None

    def get_feature_columns(self, data):
        # Exclude metadata columns
        exclude = [ID_COL, TARGET_COL, WEIGHT_COL, "survey_id"]
        return [c for c in data.columns if c not in exclude]

    def build_pipeline(self, X):
        # Filter out metadata columns from being selected
        feature_cols = self.get_feature_columns(X)
        X_subset = X[feature_cols]

        numeric_features = X_subset.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_subset.select_dtypes(include=['object', 'category']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.pipeline

    def fit_transform(self, X):
        if self.pipeline is None:
            self.build_pipeline(X)
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        return self.pipeline.transform(X)
