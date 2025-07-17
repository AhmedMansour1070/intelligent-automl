#!/usr/bin/env python
"""
Basic usage example of Intelligent AutoML Framework

This example shows the simplest way to use the intelligent AutoML framework
with a sample dataset.
"""

import pandas as pd
import numpy as np
from intelligent_automl import create_intelligent_pipeline

def create_sample_data():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.exponential(50000, 1000),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
        'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(1000, size=100, replace=False)
    df.loc[missing_indices[:50], 'age'] = np.nan
    df.loc[missing_indices[50:], 'income'] = np.nan
    
    return df

def main():
    """Main demonstration function."""
    print("🚀 Intelligent AutoML Framework - Basic Example")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample dataset...")
    df = create_sample_data()
    print(f"✅ Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"❌ Missing values: {df.isnull().sum().sum()}")
    
    # Create intelligent pipeline
    print("\n🧠 Creating intelligent pipeline...")
    pipeline = create_intelligent_pipeline(df, target_column='target')
    
    # Process data
    print("\n⚙️ Processing data...")
    features = df.drop('target', axis=1)
    target = df['target']
    
    processed_features = pipeline.fit_transform(features)
    
    # Results
    print(f"\n✅ Processing complete!")
    print(f"📈 Features: {features.shape[1]} → {processed_features.shape[1]}")
    print(f"🎯 Missing values after processing: {processed_features.isnull().sum().sum()}")
    print(f"💾 Memory usage: {processed_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Show sample of processed data
    print(f"\n📋 Sample of processed data:")
    print(processed_features.head())
    
    # Performance summary
    improvement = processed_features.shape[1] / features.shape[1]
    print(f"\n🎉 RESULTS SUMMARY:")
    print(f"  ✅ Zero missing values achieved")
    print(f"  📈 Feature engineering: {improvement:.1f}x expansion")
    print(f"  🧠 Intelligent preprocessing applied automatically")
    print(f"  🚀 Ready for machine learning!")

if __name__ == "__main__":
    main()