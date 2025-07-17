from automl_framework.intelligence.pipeline_selector import create_intelligent_pipeline
import pandas as pd
df = pd.read_csv('test.csv')


pipeline = create_intelligent_pipeline(df, target_column='trip_duration')
pipeline.fit(df.drop('trip_duration', axis=1))

intelligent_result = pipeline.transform(df.drop('trip_duration', axis=1))

print(f"  Intelligent: {intelligent_result.shape[1]} features")

print(f" Data Quality After Processing:")
print(f"  Intelligent missing values: {intelligent_result.isnull().sum().sum()}")

