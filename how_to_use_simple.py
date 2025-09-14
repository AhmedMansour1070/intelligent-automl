import os
os.environ['INTELLIGENT_AUTOML_QUIET'] = 'True'

from intelligent_automl import IntelligentAutoMLFramework


import pandas as pd

# Add this line before running the framework
df = pd.read_csv('train.csv')
# for col in df.select_dtypes(include=['category']).columns:
#     df[col] = df[col].astype(str)
# df.to_csv('test_fixed.csv', index=False)

from sklearn.preprocessing import LabelEncoder

categorical_columns = ['Pclass', 'Sex', 'Embarked']
label_encoders = {}




for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        # Handle missing values first
        df[col] = df[col].fillna('Missing')
        # Encode
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  ✅ Encoded {col}")

# Save the fixed data
df.to_csv('train_fixed.csv', index=False)


framework = IntelligentAutoMLFramework(verbose=True)  # Keep this for basic output

# Use a smaller sample for testing
results = framework.run_complete_pipeline(
    'train_fixed.csv',
    'Survived',  # Make sure this column exists in your CSV
    models_to_try=['random_forest'],  # Just one model for speed
    time_limit_minutes=1  # Quick test
)