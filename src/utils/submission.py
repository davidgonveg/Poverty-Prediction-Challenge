import pandas as pd
import math
import zipfile
from src.config import SUBMISSION_DIR

def derive_survey_id(hhid):
    # Heuristic: First digit or first few digits. 
    # Train: 100xxx, 200xxx, 300xxx
    # Test: 400xxx, 500xxx, 600xxx
    # Using integer division 100000 * 100000 to normalize?
    # Actually, 400001 // 100000 = 4. 4 * 100000 = 400000.
    return (hhid // 100000) * 100000

def generate_submission(predictions, test_features, submission_format_dir):
    # predictions: array of cons_pp
    # test_features: DataFrame with hhid and weight
    
    # 1. Create household consumption submission
    submission_cons = test_features[['hhid']].copy()
    submission_cons['per_capita_household_consumption'] = predictions
    submission_cons['survey_id'] = submission_cons['hhid'].apply(derive_survey_id)
    
    # Reorder columns as per example: survey_id,household_id,per_capita_household_consumption
    # Note: example says 'household_id' but file likely has 'hhid' or we need to rename.
    # Let's check submission_format header again later, but safe bet is to follow example exact naming if possible.
    # Example: survey_id,household_id,per_capita_household_consumption
    # Rename explicitly to match submission_format: survey_id, hhid, cons_ppp17
    submission_cons = submission_cons.rename(columns={'per_capita_household_consumption': 'cons_ppp17'})
    # Ensure column order
    submission_cons = submission_cons[['survey_id', 'hhid', 'cons_ppp17']]
    
    # Save
    cons_path = SUBMISSION_DIR / "predicted_household_consumption.csv"
    submission_cons.to_csv(cons_path, index=False)
    print(f"Saved consumption predictions to {cons_path}")

    # 2. Create poverty distribution submission
    # We need to know the thresholds.
    # We can infer them from the sample submission column names.
    sample_prob_path = submission_format_dir / "predicted_poverty_distribution.csv"
    sample_prob = pd.read_csv(sample_prob_path)
    
    threshold_cols = [c for c in sample_prob.columns if c.startswith('pct_hh_below_')]
    thresholds = [float(c.split('_')[-1]) for c in threshold_cols]
    
    # Merge weights into predictions
    # We need 'weight' from test_features.
    # test_features has 'hhid' and 'weight'
    right_df = test_features[['hhid', 'weight']]
    merged = pd.merge(submission_cons, right_df, on='hhid')
    
    survey_groups = merged.groupby('survey_id')
    
    rows = []
    
    for survey_id, group in survey_groups:
        total_weight = group['weight'].sum()
        row = {'survey_id': survey_id}
        
        for thresh_col, thresh_val in zip(threshold_cols, thresholds):
            # Sum weights where cons < thresh_val
            w_below = group[group['cons_ppp17'] < thresh_val]['weight'].sum()
            row[thresh_col] = w_below / total_weight
            
        rows.append(row)
        
    submission_prob = pd.DataFrame(rows)
    # Ensure order matches sample
    submission_prob = submission_prob[sample_prob.columns]
    
    prob_path = SUBMISSION_DIR / "predicted_poverty_distribution.csv"
    submission_prob.to_csv(prob_path, index=False)
    print(f"Saved poverty distribution predictions to {prob_path}")
    
    # Zip
    zip_path = SUBMISSION_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(cons_path, arcname=cons_path.name)
        zipf.write(prob_path, arcname=prob_path.name)
    print(f"Created submission file at {zip_path}")
