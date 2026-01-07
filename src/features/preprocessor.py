import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import ID_COL, TARGET_COL, WEIGHT_COL

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pca = None
        self.enc = None
        self.asset_cols = []
        
    def fit(self, X, y=None):
        # Select asset columns (binary/ordinal)
        candidates = ['owner', 'water', 'toilet', 'sewer', 'elect', 'roof', 'wall', 'floor']
        self.asset_cols = [c for c in candidates if c in X.columns]
        
        if self.asset_cols:
            # We must encode them first because they are strings ('Owner', 'Access'...)
            self.enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_assets = X[self.asset_cols].fillna('missing')
            X_encoded = self.enc.fit_transform(X_assets)
            
            self.pca = PCA(n_components=1)
            self.pca.fit(X_encoded)
            
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Wealth Index (PCA)
        if self.pca and self.enc and self.asset_cols:
            X_assets = X[self.asset_cols].fillna('missing')
            X_encoded = self.enc.transform(X_assets)
            X['wealth_index'] = self.pca.transform(X_encoded)[:, 0]
            
        # 2. Dietary Diversity Score
        # Consumed columns are 'Yes'/'No'
        consumed_cols = [c for c in X.columns if c.startswith('consumed')]
        if consumed_cols:
            # Create a copy to avoid SettingWithCopy if we modify
            # Map Yes->1, No->0. everything else->0
            # We use apply map for speed on subset
            def yes_no_mapper(val):
                if isinstance(val, str):
                    if val.lower() == 'yes': return 1
                return 0
                
            # Vectorized mapping is faster
            # But let's safe bet with simple replace
            # Alternatively: (X[consumed_cols] == 'Yes').sum(axis=1)
            X['dietary_diversity'] = (X[consumed_cols] == 'Yes').sum(axis=1)
            
        # 3. Household Ratios
        if 'hsize' in X.columns:
            # Force numeric
            X['hsize'] = pd.to_numeric(X['hsize'], errors='coerce').fillna(1)
            
            children_cols = [c for c in ['num_children5', 'num_children10', 'num_children18'] if c in X.columns]
            elderly_col = 'num_elderly' if 'num_elderly' in X.columns else None
            
            dependents = pd.Series(0.0, index=X.index)
            for c in children_cols:
                dependents += pd.to_numeric(X[c], errors='coerce').fillna(0)
            if elderly_col:
                dependents += pd.to_numeric(X[elderly_col], errors='coerce').fillna(0)
                
            X['dependency_ratio'] = dependents / X['hsize']
            
            # Education per capita
            if 'educ_max' in X.columns:
                 X['educ_per_capita'] = pd.to_numeric(X['educ_max'], errors='coerce').fillna(0) / X['hsize']
                 
            # Workers ratio
            if 'employed' in X.columns:
                 X['employed_ratio'] = pd.to_numeric(X['employed'], errors='coerce').fillna(0) / X['hsize']

        return X

class Preprocessor:
    def __init__(self):
        self.pipeline = None

    def get_feature_columns(self, data):
        # Exclude metadata columns
        exclude = [ID_COL, TARGET_COL, WEIGHT_COL, "survey_id", "country", "strata"]
        return [c for c in data.columns if c not in exclude]

    def fit_transform(self, X):
        # 1. Feature Engineering Step
        fe = FeatureEngineer()
        
        # We need to fit FE first to generate new columns, so we can detect them for CT
        fe.fit(X)
        X_eng = fe.transform(X)
        
        # 2. Define ColumnTransformer based on ENGINEERED data
        feature_cols = self.get_feature_columns(X_eng)
        X_subset = X_eng[feature_cols]

        numeric_features = X_subset.select_dtypes(include=['int64', 'float64']).columns
        # Important: Ensure 'wealth_index' etc are included. They are float64.
        
        categorical_features = X_subset.select_dtypes(include=['object', 'category']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        ct = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        # 3. Build Full Pipeline
        self.pipeline = Pipeline(steps=[
            ('fe', fe),
            ('ct', ct)
        ])
        
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        return self.pipeline.transform(X)
