import numpy as np
import pandas as pd

def get_poverty_thresholds(df_columns):
    # Extract thresholds from column names like 'pct_hh_below_3.17'
    threshold_cols = [c for c in df_columns if c.startswith('pct_hh_below_')]
    thresholds = [float(c.split('_')[-1]) for c in threshold_cols]
    return threshold_cols, thresholds

def calculate_weights(threshold_vals):
    # "This weighting prioritizes the poverty thresholds more the closer they are to the 
    # threshold corresponding to a 40% poverty rate"
    # The thresholds are ventiles (5%, 10%, ...). 40% is the 8th ventile (index 7).
    # Approximate rank-based weights.
    # rank of 40% is 0.40.
    
    # We map thresholds to their approx ranks (0.05, 0.10, ... 0.95)
    ranks = np.arange(0.05, 1.0, 0.05)[:len(threshold_vals)]
    
    # Simple proximity weight: 1 / (1 + |rank - 0.40|)
    # This gives max weight at 0.40.
    weights = 1.0 / (1.0 + np.abs(ranks - 0.40))
    
    # Normalize sum to len(weights)? Or just use generic weighted mean.
    return weights

def mean_absolute_percentage_error(y_true, y_pred, weights=None):
    # MAPE = mean( |(true - pred) / true| )
    # Avoid division by zero
    y_true = np.maximum(y_true, 1e-6) 
    
    errors = np.abs((y_true - y_pred) / y_true)
    
    if weights is not None:
        return np.average(errors, weights=weights)
    else:
        return np.mean(errors)

def score_submission_df(submission_df, gt_df, rates_gt_df):
    """
    Computes the official metric:
    90% * Weighted_MAPE(Rates) + 10% * MAPE(Consumption)
    """
    
    # 1. Consumption Error (10%)
    # Merge on hhid
    # Assuming submission_df has 'household_id', 'per_capita_household_consumption', 'survey_id'
    # gt_df has 'hhid', 'cons_ppp17'
    
    # Rename for merge if needed
    sub_cons = submission_df[['household_id', 'per_capita_household_consumption']].copy()
    gt_cons = gt_df[['hhid', 'cons_ppp17']].copy()
    
    merged_cons = pd.merge(sub_cons, gt_cons, left_on='household_id', right_on='hhid')
    
    cons_mape = mean_absolute_percentage_error(
        merged_cons['cons_ppp17'], 
        merged_cons['per_capita_household_consumption']
    )
    
    # 2. Poverty Rate Error (90%)
    # We need to calculate the predicted rates from the submission structure if passed as DF,
    # OR if submission_df contains the distribution CSV content.
    # Usually we generate two CSVs. Let's assume input is the distribution dataframe for rates part.
    # But `score_submission_df` might be called with the objects ready.
    
    # Let's split this function.
    return cons_mape

def calculate_poverty_metric(y_true_cons, y_pred_cons, hh_ids, weights, survey_ids, threshold_cols, thresholds):
    """
    Full metric calculation from raw consumption predictions.
    
    y_true_cons: array of true consumption
    y_pred_cons: array of predicted consumption
    hh_ids: array of household IDs
    weights: array of household weights
    survey_ids: array of survey IDs
    threshold_cols: list of column names for thresholds
    thresholds: list of float thresholds
    """
    
    # 1. Consumption MAPE (10%)
    cons_mape = mean_absolute_percentage_error(y_true_cons, y_pred_cons)
    
    # 2. Rate MAPE (90%)
    
    # We need to compute rates per survey.
    df = pd.DataFrame({
        'survey_id': survey_ids,
        'cons_pred': y_pred_cons,
        'cons_true': y_true_cons,
        'weight': weights
    })
    
    # True Rates (can actally calculate them from y_true_cons to be consistent, or use gt provided)
    # The metric compares "Predicted Rates" vs "Actual Rates".
    # Actual rates are provided in train_rates_gt.csv, OR we calculate them from true consumption.
    # Calculating from true consumption is safer for CV splits where we might not have the full survey GT pre-calculated.
    
    rate_mape_scores = []
    
    metric_weights = calculate_weights(thresholds)
    
    for survey in df['survey_id'].unique():
        grp = df[df['survey_id'] == survey]
        total_weight = grp['weight'].sum()
        
        # Calculate Pred Rates and True Rates for this subset
        # This handles the case where we validate on a Fold (subset of survey) or a full survey.
        # IF VALIDATING ON FOLDS: The distribution of the Fold might differ from the full survey.
        # But standard CV implies we check error on the validation set.
        
        pred_rates = []
        true_rates = []
        
        for t in thresholds:
            # Pred
            w_below_pred = grp[grp['cons_pred'] < t]['weight'].sum()
            pred_rates.append(w_below_pred / total_weight)
            
            # True
            w_below_true = grp[grp['cons_true'] < t]['weight'].sum()
            true_rates.append(w_below_true / total_weight)
            
        # MAPE for this survey/fold
        survey_mape = mean_absolute_percentage_error(np.array(true_rates), np.array(pred_rates), weights=metric_weights)
        rate_mape_scores.append(survey_mape)
        
    avg_rate_mape = np.mean(rate_mape_scores)
    
    final_score = 0.9 * avg_rate_mape + 0.1 * cons_mape
    return final_score, avg_rate_mape, cons_mape
