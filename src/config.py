from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_FEATURES = DATA_DIR / "train_hh_features.csv"
TRAIN_LABELS = DATA_DIR / "train_hh_gt.csv"
TRAIN_RATES = DATA_DIR / "train_rates_gt.csv"
TEST_FEATURES = DATA_DIR / "test_hh_features.csv"
SUBMISSION_DIR = DATA_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

# Columns
ID_COL = "hhid"
SURVEY_COL = "strata" # Assuming strata identifies the survey or region roughly, closer inspection needed. Wait, problem says "three surveys (IDs 100000, 200000 and 300000)". 
# Actually, looking at the data head from previous steps:
# train_features: hhid, weight, strata... 
# train_gt: survey_id, hhid, cons_pp (No, wait. Previous output showed: "hhid,com,weight,strata" for features, and "100004,1,375,4,824.617" which is confusing.
# Let's re-verify column names in data loader step. 
# BUT based on standard competition formats:
SURVEY_ID_COL = "country" # Or similar. Let's start with 'strata' as placeholder or check headers again.
# Wait, let's look at the previous `head` output for `train_hh_gt.csv`:
# "survey_id,hhid,cons_pp" -> So survey_id is explicit there.
# For `train_hh_features.csv`, the head was: "hhid,com,weight,strata". 
# Wait, "com" looks like it might serve as survey_id or similar? 
# The text says "surveys (IDs 100000, 200000 and 300000)". 
# Let's assume there is a column for survey_id. 
# I will output a config with placeholders and refine it after a quick check if needed.
# Actually, I should use `run_command` to inspect headers again to be 100% sure for config.

TARGET_COL = "cons_ppp17"
WEIGHT_COL = "weight"

# Random Seed
RANDOM_SEED = 42

MODEL_PATH = "models/baseline.pkl"
