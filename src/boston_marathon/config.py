"""Paths, schema, checkpoint labels, and temporal splits."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_RESULTS_CSV = PROJECT_ROOT / 'data' / 'processed' / 'boston_marathon_cleaned.csv'
PIPELINE_METRICS_JSON = PROJECT_ROOT / 'results' / 'pipeline_metrics.json'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'

CUMULATIVE_SPLIT_TIME_COLUMNS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds', 'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']
CHECKPOINT_LABELS = [c.replace('_seconds', '').upper() for c in CUMULATIVE_SPLIT_TIME_COLUMNS]

PRE_RACE_REGRESSION_TRAIN_YEARS = range(2000, 2017)
PRE_RACE_REGRESSION_TEST_YEARS = (2017,)
RUNNER_MIXED_EFFECTS_TRAIN_END_YEAR = 2016
IN_RACE_SPLIT_PREDICTION_TRAIN_YEARS = (2015, 2016)
IN_RACE_SPLIT_PREDICTION_TEST_YEAR = 2017

YEAR_CENTER = 2010

PRE_RACE_FIXED_EFFECT_FEATURES = [
    'age_centered', 'age_centered_squared', 'female', 'year_centered',
    'age_centered_female_interaction', 'age_centered_squared_female_interaction',
]
