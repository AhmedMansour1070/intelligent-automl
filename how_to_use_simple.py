import os
os.environ['INTELLIGENT_AUTOML_QUIET'] = 'True'

from intelligent_automl import IntelligentAutoMLFramework


import pandas as pd

# Add this line before running the framework
df = pd.read_csv('test.csv')
for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].astype(str)
df.to_csv('test_fixed.csv', index=False)


framework = IntelligentAutoMLFramework(verbose=True)  # Keep this for basic output

# Use a smaller sample for testing
results = framework.run_complete_pipeline(
    'test_fixed.csv',
    'trip_duration',  # Make sure this column exists in your CSV
    models_to_try=['random_forest'],  # Just one model for speed
    time_limit_minutes=1  # Quick test
)