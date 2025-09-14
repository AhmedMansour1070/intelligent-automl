from intelligent_automl import IntelligentAutoMLFramework
import pandas as pd
# Initialize framework
framework = IntelligentAutoMLFramework(verbose=True)
df_small = pd.read_csv('train.csv')
# drop ["PassengerId", "Name", "Ticket", "Cabin"]
df_small = df_small.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
# upload the small dataset as a csv file
df_small.to_csv('small_train.csv')

# Run complete pipeline on your taxi data
results = framework.run_complete_pipeline(
    'small_train.csv',
    'Survived',
    output_dir='results'
    )

# Get your trained model
best_model = results['results']['model_training']['best_model']
best_score = results['results']['model_training']['best_score']
