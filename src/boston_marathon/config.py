"""Central configuration for the entire pipeline.

Every other module in this package imports paths and constants from here,
so changing a value here propagates everywhere automatically. The three
groups of settings correspond to the three stages of the analysis:

  - File paths       data.py reads CLEANED_CSV; rq2.py writes/reads per-runner predictions
  - Checkpoint info   rq3.py uses SPLIT_COLS, CHECKPOINT_KM, and model labels
  - Year boundaries   each RQ uses its own train/test split to prevent leakage
  - Weather data      rq2.py and weather.py use temperatures and race dates
"""
from pathlib import Path

# File paths (resolved from project root)
_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED_CSV = _ROOT / 'data' / 'processed' / 'boston_marathon_cleaned.csv'
BLUP_CSV = _ROOT / 'data' / 'processed' / 'runner_blups.csv'
BLUP_LEAKFREE_CSV = _ROOT / 'data' / 'processed' / 'runner_blups_leakfree.csv'
WEATHER_CACHE = _ROOT / 'data' / 'processed' / 'race_weather_cache.json'

# Marathon course checkpoints (used by rq3.py)
SPLIT_COLS = ['5k_seconds', '10k_seconds', '15k_seconds', '20k_seconds',
              'half_seconds', '25k_seconds', '30k_seconds', '35k_seconds', '40k_seconds']
CHECKPOINT_KM = [5.0, 10.0, 15.0, 20.0, 21.0975, 25.0, 30.0, 35.0, 40.0]
MARATHON_KM = 42.195
CP_ORDER = [c.replace('_seconds', '').upper() for c in SPLIT_COLS]

# RQ3 model display labels (one per variant in run_progressive)
NAIVE = 'Naive (constant pace)'
SPLITS = 'Ridge (splits only)'
DEMO = 'Ridge (splits + demographics)'
FULL = 'Ridge (splits + demo + history)'
SPLITS_SUBSET = 'Ridge splits (history subset)'
SINGLE = 'Ridge (single checkpoint)'
DEMO_YEAR = 'Ridge (splits + demo + year)'

# Temporal hold-out boundaries (each RQ has its own to prevent leakage)
RQ1_TRAIN_YEARS = range(2000, 2018)    # test on 2018-2019
RQ1_TEST_YEARS = (2018, 2019)
RQ2_HOLDOUT_MAX_YEAR = 2016            # test on 2017-2019
RQ3_TRAIN_YEARS = (2015, 2016)         # test on 2017 (only years with splits)
RQ3_TEST_YEAR = 2017

# Race-day weather (used by rq2._add_weather_cols and weather.py)
# Finish-line temperature in Fahrenheit from the John Hancock Marathon Weather History PDF
RACE_TEMP_F = {2000:47, 2001:54, 2002:56, 2003:59, 2004:86, 2005:66, 2006:53, 2007:50,
               2008:53, 2009:47, 2010:55, 2011:55, 2012:87, 2013:54, 2014:62, 2015:46,
               2016:62, 2017:73, 2018:46, 2019:63}
# Race dates (Patriots' Day) — weather.py queries Open-Meteo for humidity and wind
RACE_DATES = {
    2000:'2000-04-17', 2001:'2001-04-16', 2002:'2002-04-15', 2003:'2003-04-21',
    2004:'2004-04-19', 2005:'2005-04-18', 2006:'2006-04-17', 2007:'2007-04-16',
    2008:'2008-04-21', 2009:'2009-04-20', 2010:'2010-04-19', 2011:'2011-04-18',
    2012:'2012-04-16', 2013:'2013-04-15', 2014:'2014-04-21', 2015:'2015-04-20',
    2016:'2016-04-18', 2017:'2017-04-17', 2018:'2018-04-16', 2019:'2019-04-15',
}

# Shared constants
PERSONALIZED_RMSE = 996.2   # Random Intercept + Slope in-sample conditional RMSE; crossover benchmark in rq3.py
YEAR_CENTER = 2010           # centering point for the year covariate
