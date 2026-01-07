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
        self.pca_cons = None
        self.context_maps = {}
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
            
            # Generate wealth index strictly for internal aggregation calculation
            wealth_index = self.pca.transform(X_encoded)[:, 0]
        else:
            wealth_index = np.zeros(len(X))

        # --- Consumption PCA ---
        consumed_cols = [c for c in X.columns if c.startswith('consumed')]
        if consumed_cols:
            X_cons = (X[consumed_cols] == 'Yes').astype(int)
            self.pca_cons = PCA(n_components=3)
            self.pca_cons.fit(X_cons)
            
        # --- Context Aggregates ---
        # We need to construct temporary df to groupby
        temp_df = X.copy()
        temp_df['wealth_index'] = wealth_index
        
        # Ensure numeric for aggregation targets
        if 'hsize' in temp_df.columns:
             temp_df['hsize'] = pd.to_numeric(temp_df['hsize'], errors='coerce')
        
        # Education Mapping
        if 'educ_max' in temp_df.columns:
             # Standardize to years (Approximate)
             edu_map = {
                 'Never attended': 0,
                 'Incomplete Primary': 3,
                 'Complete Primary': 6,
                 'Incomplete Secondary': 9,
                 'Complete Secondary': 12,
                 'Incomplete University': 14,
                 'Complete University': 16,
                 'Incomplete Technical': 14,
                 'Complete Technical': 15,
                 'Post-graduate': 18
             }
             # Map and coerce any failures to median or 0
             # We use map then fillna with 0
             temp_df['mapped_educ'] = temp_df['educ_max'].map(edu_map).fillna(0)
             self.edu_map_ref = edu_map # Store for transform
             # Use the mapped column for aggregation
             temp_df['educ_max'] = temp_df['mapped_educ']
             
        # Reconstruct Region
        region_cols = [c for c in X.columns if c.startswith('region')]
        if region_cols:
            temp_df['region_id'] = temp_df[region_cols].idxmax(axis=1)
            
            # Calculate Means
            # We agg by Region + Urban (if available? Urban is 0/1 usually or string?)
            # Let's stick to Region for robustness (Region includes Urban/Rural mix mostly)
            # Actually dataset has 'urban' column. Grouping by ['region_id', 'urban'] is better.
            
            group_cols = ['region_id']
            if 'urban' in temp_df.columns:
                group_cols.append('urban')
                
            agg_targets = ['wealth_index', 'hsize', 'educ_max']
            agg_targets = [t for t in agg_targets if t in temp_df.columns]
            
            # Create a combined key or just map using tuple? a bit complex for pandas map
            # Simpler: Just Region for now to avoid sparsity
            means = temp_df.groupby('region_id')[agg_targets].mean()
            
            self.context_maps = {}
            for t in agg_targets:
                self.context_maps[t] = means[t]
            
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Wealth Index (PCA)
        if self.pca and self.enc and self.asset_cols:
            X_assets = X[self.asset_cols].fillna('missing')
            X_encoded = self.enc.transform(X_assets)
            X['wealth_index'] = self.pca.transform(X_encoded)[:, 0]
            
        # 2. Consumption Pattern PCA (Lifestyle)
        # Consumed columns are 'Yes'/'No'
        consumed_cols = [c for c in X.columns if c.startswith('consumed')]
        if consumed_cols:
            # First, convert to binary (0/1) locally
            # We can use vectorized because values are likely homogenous strings
            # But let's be safe
            X_cons = (X[consumed_cols] == 'Yes').astype(int)
            
            # Simple Sum (Dietary Diversity)
            X['dietary_diversity'] = X_cons.sum(axis=1)
            
            # PCA on Consumption (if fitted)
            if self.pca_cons:
                 cons_pca = self.pca_cons.transform(X_cons)
                 X['cons_comp1'] = cons_pca[:, 0]
                 X['cons_comp2'] = cons_pca[:, 1]
                 X['cons_comp3'] = cons_pca[:, 2]

        # 3. Context Features (Aggregates by Region)
        # Reconstruct Region ID from One-Hot
        region_cols = [c for c in X.columns if c.startswith('region')]
        if region_cols:
            # argmax to get 0, 1, 2... representing region
            # We add a temp 'region_id' column
            X['region_id'] = X[region_cols].idxmax(axis=1)
            
            # Apply Aggregates (if fitted)
            if self.context_maps:
                # Map using index
                for feature, mapping in self.context_maps.items():
                    # mapping is a dict/series: region_id -> mean_value
                    # We map X['region_id'] to get the mean
                    mean_val = X['region_id'].map(mapping)
                    X[f'mean_{feature}_by_region'] = mean_val
                    # Relative Feature: My Value / Regional Mean
                    # Avoid division by zero
                    if feature in X.columns:
                         # Handle educ_max specially for relative calculation
                         if feature == 'educ_max':
                             current_val = X[feature].map(self.edu_map_ref).fillna(0) if hasattr(self, 'edu_map_ref') else pd.to_numeric(X[feature], errors='coerce').fillna(0)
                         else:
                             current_val = X[feature]
                             
                         X[f'relative_{feature}'] = current_val / (mean_val + 1e-6)

        # 4. Household Ratios
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
