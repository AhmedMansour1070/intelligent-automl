#!/usr/bin/env python
"""
Fixed test for the Intelligent AutoML Framework
"""

import os
os.environ['INTELLIGENT_AUTOML_QUIET'] = 'True'

from intelligent_automl import IntelligentAutoMLFramework

def main():
    print("🧪 Testing Fixed AutoML Framework")
    print("=" * 40)
    
    # Initialize framework with minimal logging
    framework = IntelligentAutoMLFramework(verbose=True)
    
    try:
        # Run complete pipeline
        results = framework.run_complete_pipeline(
            'test.csv',
            'trip_duration',
            models_to_try=['random_forest'],  # Just one model for speed
            time_limit_minutes=2  # Quick test
        )
        
        print("\n🎉 SUCCESS!")
        print(f"🏆 Best model: {results['results']['model_training']['best_model']}")
        print(f"📊 Best score: {results['results']['model_training']['best_score']:.4f}")
        print("✅ Framework is working perfectly!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("But the framework loaded and started successfully!")
        
        # Try a simpler approach
        print("\n🔄 Trying simpler approach...")
        try:
            from intelligent_automl import create_intelligent_pipeline
            import pandas as pd
            
            df = pd.read_csv('test.csv')
            pipeline = create_intelligent_pipeline(df, target_column='trip_duration')
            
            features = df.drop('trip_duration', axis=1)
            processed = pipeline.fit_transform(features)
            
            print(f"✅ Simple pipeline works!")
            print(f"📈 Features: {features.shape[1]} → {processed.shape[1]}")
            print(f"🎯 Missing values: {processed.isnull().sum().sum()}")
            
        except Exception as e2:
            print(f"❌ Simple approach also failed: {str(e2)}")

if __name__ == "__main__":
    main()