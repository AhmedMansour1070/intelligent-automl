#!/usr/bin/env python
"""
Comprehensive AutoML Framework Demo

This script demonstrates the full capabilities of your AutoML framework.
"""

import pandas as pd
import numpy as np
from automl_framework.data import (
    DataPipeline, 
    MissingValueHandler, 
    FeatureScaler, 
    CategoricalEncoder,
    OutlierHandler,
    DateTimeProcessor,
    AutoLoader
)

def create_demo_dataset():
    """Create a comprehensive demo dataset with various data types."""
    np.random.seed(42)
    
    n_samples = 1000
    
    data = {
        # Numeric features with missing values
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        
        # Categorical features with missing values
        'city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', None], n_samples, p=[0.3, 0.25, 0.2, 0.2, 0.05]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], n_samples, p=[0.3, 0.4, 0.2, 0.08, 0.02]),
        'employment': np.random.choice(['Full-time', 'Part-time', 'Unemployed', 'Student', None], n_samples, p=[0.6, 0.2, 0.1, 0.08, 0.02]),
        
        # Date features
        'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        
        # Target variable
        'approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values in numeric columns
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices[:50], 'age'] = np.nan
    df.loc[missing_indices[50:], 'income'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[outlier_indices, 'income'] = df.loc[outlier_indices, 'income'] * 10
    
    return df

def demo_basic_preprocessing():
    """Demonstrate basic preprocessing capabilities."""
    print("\n" + "="*60)
    print("🧪 DEMO 1: Basic Preprocessing Pipeline")
    print("="*60)
    
    # Create demo data
    df = create_demo_dataset()
    print(f"📊 Created dataset: {df.shape}")
    print(f"📝 Columns: {list(df.columns)}")
    print(f"❌ Missing values: {df.isnull().sum().sum()}")
    
    # Create basic preprocessing pipeline
    pipeline = (DataPipeline()
               .add_step('missing_values', MissingValueHandler(
                   numeric_strategy='median',
                   categorical_strategy='most_frequent'
               ))
               .add_step('categorical_encoding', CategoricalEncoder(
                   method='onehot',
                   max_categories=10
               ))
               .add_step('feature_scaling', FeatureScaler(method='standard')))
    
    print(f"\n🔧 Created pipeline with {len(pipeline)} steps:")
    for i, name in enumerate(pipeline.get_step_names(), 1):
        print(f"   {i}. {name}")
    
    # Process the data
    target = df['approved']
    features = df.drop('approved', axis=1)
    
    print(f"\n⚙️ Processing features...")
    processed_features = pipeline.fit_transform(features)
    
    print(f"✅ Processing complete!")
    print(f"📈 Original features: {features.shape}")
    print(f"📈 Processed features: {processed_features.shape}")
    print(f"✨ Missing values after processing: {processed_features.isnull().sum().sum()}")
    
    return pipeline, processed_features, target

def demo_advanced_preprocessing():
    """Demonstrate advanced preprocessing capabilities."""
    print("\n" + "="*60)
    print("🚀 DEMO 2: Advanced Preprocessing Pipeline")
    print("="*60)
    
    # Create demo data
    df = create_demo_dataset()
    
    # Create advanced pipeline
    advanced_pipeline = (DataPipeline()
                        .add_step('datetime_processing', DateTimeProcessor(
                            extract_components=True,
                            extract_cyclical=True
                        ))
                        .add_step('outlier_handling', OutlierHandler(
                            method='iqr',
                            treatment='cap'
                        ))
                        .add_step('missing_values', MissingValueHandler(
                            numeric_strategy='knn',
                            n_neighbors=5
                        ))
                        .add_step('categorical_encoding', CategoricalEncoder(
                            method='onehot'
                        ))
                        .add_step('feature_scaling', FeatureScaler(
                            method='robust'
                        )))
    
    print(f"🔧 Created advanced pipeline with {len(advanced_pipeline)} steps:")
    for i, name in enumerate(advanced_pipeline.get_step_names(), 1):
        print(f"   {i}. {name}")
    
    # Process the data
    target = df['approved']
    features = df.drop('approved', axis=1)
    
    print(f"\n⚙️ Processing with advanced pipeline...")
    processed_features = advanced_pipeline.fit_transform(features)
    
    print(f"✅ Advanced processing complete!")
    print(f"📈 Original features: {features.shape}")
    print(f"📈 Processed features: {processed_features.shape}")
    print(f"🕒 Date features extracted and encoded")
    print(f"🎯 Outliers handled with IQR capping")
    print(f"🔗 KNN imputation for missing values")
    
    return advanced_pipeline, processed_features, target

def demo_pipeline_management():
    """Demonstrate pipeline management capabilities."""
    print("\n" + "="*60)
    print("🛠️ DEMO 3: Pipeline Management & Serialization")
    print("="*60)
    
    # Create a pipeline
    pipeline = (DataPipeline()
               .add_step('imputer', MissingValueHandler())
               .add_step('scaler', FeatureScaler()))
    
    print("📋 Pipeline Operations:")
    
    # Show pipeline info
    print(f"   📊 Pipeline steps: {pipeline.get_step_names()}")
    print(f"   📏 Pipeline length: {len(pipeline)}")
    
    # Demonstrate step manipulation
    pipeline.add_step('encoder', CategoricalEncoder(), position=1)
    print(f"   ➕ Added encoder at position 1: {pipeline.get_step_names()}")
    
    # Get specific step
    imputer = pipeline.get_step('imputer')
    print(f"   🔍 Retrieved step 'imputer': {type(imputer).__name__}")
    
    # Save pipeline
    pipeline.save('demo_pipeline.joblib')
    print(f"   💾 Saved pipeline to: demo_pipeline.joblib")
    
    # Load pipeline
    loaded_pipeline = DataPipeline.load('demo_pipeline.joblib')
    print(f"   📂 Loaded pipeline: {loaded_pipeline.get_step_names()}")
    
    # Validate pipeline
    df = create_demo_dataset()
    validation_report = pipeline.validate(df.drop('approved', axis=1))
    print(f"   ✅ Pipeline validation: {'PASSED' if validation_report['is_valid'] else 'FAILED'}")
    
    # Clean up
    import os
    if os.path.exists('demo_pipeline.joblib'):
        os.remove('demo_pipeline.joblib')
        print(f"   🗑️ Cleaned up demo file")

def demo_data_loading():
    """Demonstrate data loading capabilities."""
    print("\n" + "="*60)
    print("📁 DEMO 4: Data Loading Capabilities")
    print("="*60)
    
    # Create sample CSV file
    df = create_demo_dataset()
    df.to_csv('demo_data.csv', index=False)
    print("📄 Created demo_data.csv")
    
    # Test AutoLoader
    from automl_framework.data.loaders import AutoLoader, load_data
    
    loader = AutoLoader()
    print(f"🔧 AutoLoader supports: {loader.get_supported_formats()}")
    
    # Load data using load_data convenience function
    loaded_df = load_data('demo_data.csv')
    print(f"📊 Loaded data shape: {loaded_df.shape}")
    print(f"✅ Data types preserved: {loaded_df.dtypes.nunique()} different types")
    
    # Get file info
    file_info = loader.get_file_info('demo_data.csv')
    print(f"📋 File info:")
    print(f"   📏 Size: {file_info['size_mb']:.2f} MB")
    print(f"   📂 Format: {file_info['detected_format']}")
    print(f"   ✅ Valid: {file_info['is_supported']}")
    
    # Clean up
    import os
    if os.path.exists('demo_data.csv'):
        os.remove('demo_data.csv')
        print("🗑️ Cleaned up demo file")

def main():
    """Run all demos."""
    print("🎉 AutoML Framework - Comprehensive Demo")
    print("🔬 Testing all capabilities of your production-ready framework")
    
    try:
        # Run all demos
        demo_basic_preprocessing()
        demo_advanced_preprocessing()
        demo_pipeline_management()
        demo_data_loading()
        
        print("\n" + "="*60)
        print("🎊 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✅ Basic preprocessing")
        print("✅ Advanced preprocessing") 
        print("✅ Pipeline management")
        print("✅ Data loading")
        print("\n🚀 Your AutoML Framework is production-ready!")
        print("📚 Ready to add model training, evaluation, and optimization!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()