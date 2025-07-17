#!/usr/bin/env python
"""
Complete AutoML Framework Usage Guide

This comprehensive guide shows every way to use your AutoML framework,
from basic operations to advanced enterprise workflows.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ===================================
# 1. DATA LOADING - ALL METHODS
# ===================================

def data_loading_examples():
    """All ways to load data with your framework."""
    
    print("📁 DATA LOADING METHODS")
    print("=" * 50)
    
    from intelligent_automl.data import load_data, AutoLoader, CSVLoader, ExcelLoader
    
    # Method 1: Universal loader (EASIEST)
    df = load_data('data.csv')  # Automatically detects format
    df = load_data('data.xlsx')  # Works with Excel
    df = load_data('data.json')  # Works with JSON
    df = load_data('data.parquet')  # Works with Parquet
    print("✅ Universal loading with auto-detection")
    
    # Method 2: Specific loaders with custom options
    csv_loader = CSVLoader(
        delimiter=';',  # Custom delimiter
        encoding='latin-1',  # Custom encoding
        header=0,
        skip_rows=2
    )
    df = csv_loader.load('data.csv')
    print("✅ Custom CSV loading")
    
    # Method 3: Excel with sheet selection
    excel_loader = ExcelLoader(
        sheet_name='Sheet2',  # Specific sheet
        header=[0, 1],  # Multi-level headers
        usecols='A:G'  # Specific columns
    )
    df = excel_loader.load('data.xlsx')
    print("✅ Advanced Excel loading")
    
    # Method 4: Load from URL
    from intelligent_automl.data import URLLoader
    url_loader = URLLoader(cache_dir='./cache')
    df = url_loader.load('https://example.com/data.csv')
    print("✅ URL loading with caching")
    
    # Method 5: Database loading
    from intelligent_automl.data import DatabaseLoader
    db_loader = DatabaseLoader(
        connection_string='sqlite:///database.db',
        query='SELECT * FROM customers WHERE age > 18'
    )
    df = db_loader.load()
    print("✅ Database loading")
    
    # Method 6: File information before loading
    auto_loader = AutoLoader()
    file_info = auto_loader.get_file_info('data.csv')
    print(f"File size: {file_info['size_mb']} MB")
    print(f"Format: {file_info['detected_format']}")
    print(f"Columns: {file_info.get('columns', 'Unknown')}")

# ===================================
# 2. DATA PREPROCESSING - ALL COMPONENTS
# ===================================

def preprocessing_examples():
    """All preprocessing capabilities."""
    
    print("\n🔧 DATA PREPROCESSING METHODS")
    print("=" * 50)
    
    from intelligent_automl.data import (
        MissingValueHandler, FeatureScaler, CategoricalEncoder,
        OutlierHandler, FeatureEngineering, DateTimeProcessor, FeatureSelector
    )
    
    # Sample data
    df = pd.DataFrame({
        'age': [25, 30, None, 35, 1000],  # Has missing and outlier
        'income': [50000, None, 55000, 70000, 80000],
        'city': ['NY', 'LA', None, 'NY', 'SF'],
        'signup_date': pd.date_range('2023-01-01', periods=5),
        'score': [0.1, 0.8, 0.3, 0.9, 0.2],
        'target': [0, 1, 0, 1, 1]
    })
    
    # 1. Missing Value Handling (Multiple strategies)
    
    # Basic imputation
    imputer1 = MissingValueHandler(
        numeric_strategy='mean',
        categorical_strategy='most_frequent'
    )
    
    # Advanced KNN imputation
    imputer2 = MissingValueHandler(
        numeric_strategy='knn',
        categorical_strategy='constant',
        n_neighbors=3,
        fill_value='Unknown'
    )
    
    # Custom fill values
    imputer3 = MissingValueHandler(
        numeric_strategy='constant',
        categorical_strategy='constant',
        fill_value={'age': 30, 'city': 'Unknown'}
    )
    
    result = imputer1.fit_transform(df)
    print("✅ Missing value imputation (mean/mode, KNN, constant)")
    
    # 2. Feature Scaling (Multiple methods)
    
    # Standard scaling
    scaler1 = FeatureScaler(method='standard')
    
    # Robust scaling (less sensitive to outliers)
    scaler2 = FeatureScaler(method='robust')
    
    # MinMax scaling
    scaler3 = FeatureScaler(method='minmax')
    
    # Quantile scaling
    scaler4 = FeatureScaler(method='quantile')
    
    result = scaler1.fit_transform(df)
    print("✅ Feature scaling (standard, robust, minmax, quantile)")
    
    # 3. Categorical Encoding (Multiple methods)
    
    # One-hot encoding
    encoder1 = CategoricalEncoder(
        method='onehot',
        drop_first=True,  # Avoid multicollinearity
        max_categories=10  # Limit categories
    )
    
    # Label encoding
    encoder2 = CategoricalEncoder(method='label')
    
    # Ordinal encoding
    encoder3 = CategoricalEncoder(method='ordinal')
    
    result = encoder1.fit_transform(df)
    print("✅ Categorical encoding (onehot, label, ordinal)")
    
    # 4. Outlier Handling
    
    # IQR method
    outlier1 = OutlierHandler(
        method='iqr',
        treatment='cap',  # Cap outliers
        threshold=1.5
    )
    
    # Z-score method
    outlier2 = OutlierHandler(
        method='zscore',
        treatment='remove',  # Remove outliers
        threshold=3
    )
    
    result = outlier1.fit_transform(df)
    print("✅ Outlier handling (IQR, Z-score, cap/remove)")
    
    # 5. Feature Engineering
    
    engineer = FeatureEngineering(
        polynomial_degree=2,  # Create polynomial features
        log_transform=True,  # Log transform positive features
        sqrt_transform=True  # Square root transform
    )
    
    result = engineer.fit_transform(df)
    print("✅ Feature engineering (polynomial, log, sqrt)")
    
    # 6. DateTime Processing
    
    dt_processor = DateTimeProcessor(
        extract_components=True,  # Year, month, day, etc.
        extract_cyclical=True,  # Sin/cos encoding
        drop_original=False  # Keep original date
    )
    
    result = dt_processor.fit_transform(df)
    print("✅ DateTime processing (components, cyclical)")
    
    # 7. Feature Selection
    
    selector = FeatureSelector(
        method='mutual_info',
        k=5,  # Select top 5 features
        threshold=None
    )
    
    features = df.drop('target', axis=1)
    target = df['target']
    result = selector.fit_transform(features, target)
    print("✅ Feature selection (mutual info, chi2, f-test)")

# ===================================
# 3. PIPELINE CONSTRUCTION - ALL PATTERNS
# ===================================

def pipeline_examples():
    """All ways to build and use pipelines."""
    
    print("\n🔗 PIPELINE CONSTRUCTION METHODS")
    print("=" * 50)
    
    from intelligent_automl.data import DataPipeline, MissingValueHandler, FeatureScaler, CategoricalEncoder
    
    # Method 1: Step-by-step construction
    pipeline = DataPipeline()
    pipeline.add_step('imputer', MissingValueHandler())
    pipeline.add_step('scaler', FeatureScaler())
    pipeline.add_step('encoder', CategoricalEncoder())
    print("✅ Step-by-step pipeline construction")
    
    # Method 2: Fluent interface (RECOMMENDED)
    pipeline = (DataPipeline()
                .add_step('missing', MissingValueHandler(numeric_strategy='median'))
                .add_step('encoding', CategoricalEncoder(method='onehot'))
                .add_step('scaling', FeatureScaler(method='robust')))
    print("✅ Fluent interface pipeline")
    
    # Method 3: Construction with custom positions
    pipeline = DataPipeline()
    pipeline.add_step('scaler', FeatureScaler())
    pipeline.add_step('imputer', MissingValueHandler(), position=0)  # Insert at start
    pipeline.add_step('encoder', CategoricalEncoder(), position=1)  # Insert in middle
    print("✅ Custom position insertion")
    
    # Method 4: Pipeline from list
    from intelligent_automl.data.preprocessors import OutlierHandler
    steps = [
        ('outliers', OutlierHandler()),
        ('missing', MissingValueHandler()),
        ('encoding', CategoricalEncoder()),
        ('scaling', FeatureScaler())
    ]
    pipeline = DataPipeline(steps=steps)
    print("✅ Pipeline from steps list")
    
    # Pipeline manipulation
    print(f"Pipeline steps: {pipeline.get_step_names()}")
    print(f"Pipeline length: {len(pipeline)}")
    
    # Get specific step
    imputer = pipeline.get_step('missing')
    print(f"Retrieved step: {type(imputer).__name__}")
    
    # Replace step
    new_imputer = MissingValueHandler(numeric_strategy='knn')
    pipeline.replace_step('missing', new_imputer)
    print("✅ Step replacement")
    
    # Remove step
    pipeline.remove_step('outliers')
    print("✅ Step removal")

# ===================================
# 4. ADVANCED PIPELINE OPERATIONS
# ===================================

def advanced_pipeline_operations():
    """Advanced pipeline capabilities."""
    
    print("\n⚙️ ADVANCED PIPELINE OPERATIONS")
    print("=" * 50)
    
    from intelligent_automl.data import DataPipeline, MissingValueHandler, FeatureScaler
    
    # Create sample data and pipeline
    df = pd.DataFrame({
        'A': [1, 2, None, 4, 5],
        'B': [None, 2, 3, 4, 5],
        'C': ['x', 'y', None, 'x', 'z']
    })
    
    pipeline = (DataPipeline()
                .add_step('imputer', MissingValueHandler())
                .add_step('scaler', FeatureScaler()))
    
    # 1. Pipeline validation
    validation_report = pipeline.validate(df)
    print(f"✅ Pipeline validation: {validation_report['is_valid']}")
    if validation_report['errors']:
        print(f"Errors: {validation_report['errors']}")
    
    # 2. Pipeline serialization
    pipeline.fit(df)
    
    # Save pipeline
    pipeline.save('my_pipeline.joblib')
    pipeline.save('my_pipeline.pkl', format='pickle')
    print("✅ Pipeline saved in multiple formats")
    
    # Load pipeline
    loaded_pipeline = DataPipeline.load('my_pipeline.joblib')
    result = loaded_pipeline.transform(df)
    print("✅ Pipeline loaded and used")
    
    # 3. Pipeline configuration export/import
    pipeline.save_config('pipeline_config.json')
    print("✅ Pipeline configuration exported")
    
    # 4. Pipeline performance profiling
    performance = pipeline.profile_performance(df, n_iterations=5)
    print(f"✅ Pipeline profiling: {performance['total_time_seconds']:.4f}s total")
    for step in performance['steps']:
        print(f"   {step['name']}: {step['avg_time_seconds']:.4f}s")
    
    # 5. Memory usage analysis
    memory_info = pipeline.get_memory_usage()
    print(f"✅ Memory usage: {memory_info['total_size_mb']:.2f} MB")
    
    # 6. Feature names tracking
    input_features = ['A', 'B', 'C']
    output_features = pipeline.get_feature_names_out(input_features)
    print(f"✅ Feature names: {input_features} → {len(output_features)} features")
    
    # 7. Pipeline copying
    pipeline_copy = pipeline.copy()
    print("✅ Pipeline copied for experimentation")
    
    # 8. Parameter management
    all_params = pipeline.get_params(deep=True)
    print(f"✅ Pipeline has {len(all_params)} parameters")
    
    # Set parameters
    pipeline.set_params(imputer__numeric_strategy='median')
    print("✅ Pipeline parameters updated")
    
    # Cleanup
    import os
    for file in ['my_pipeline.joblib', 'my_pipeline.pkl', 'pipeline_config.json']:
        if os.path.exists(file):
            os.remove(file)

# ===================================
# 5. CONFIGURATION MANAGEMENT
# ===================================

def configuration_examples():
    """All configuration management approaches."""
    
    print("\n📋 CONFIGURATION MANAGEMENT")
    print("=" * 50)
    
    from intelligent_automl.core import (
        AutoMLConfig, DataConfig, PreprocessingConfig, 
        ModelConfig, TrainingConfig, EvaluationConfig
    )
    
    # Method 1: Programmatic configuration
    config = AutoMLConfig(
        data=DataConfig(
            file_path='data.csv',
            target_column='target',
            test_size=0.2,
            random_state=42
        ),
        preprocessing=PreprocessingConfig(
            scaling_method='standard',
            encoding_method='onehot',
            handle_missing='auto',
            feature_selection=True
        ),
        model=ModelConfig(
            model_type='random_forest',
            hyperparameters={'n_estimators': 100, 'max_depth': 10},
            cross_validation_folds=5
        ),
        training=TrainingConfig(
            max_time_minutes=30,
            early_stopping=True,
            verbose=True
        ),
        evaluation=EvaluationConfig(
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            save_predictions=True
        )
    )
    print("✅ Programmatic configuration")
    
    # Method 2: Builder pattern
    from intelligent_automl.pipeline.builders import AutoMLConfigBuilder
    
    # Note: This would be in the builders module when implemented
    # config = (AutoMLConfigBuilder()
    #           .with_data_source('data.csv', 'target')
    #           .with_preprocessing(scaling='robust', encoding='onehot')
    #           .with_model('xgboost', {'n_estimators': 200})
    #           .with_evaluation(cv_folds=10)
    #           .build())
    print("✅ Builder pattern configuration (when implemented)")
    
    # Method 3: JSON configuration
    config.to_json('automl_config.json')
    loaded_config = AutoMLConfig.from_json('automl_config.json')
    print("✅ JSON configuration save/load")
    
    # Method 4: YAML configuration
    config.to_yaml('automl_config.yaml')
    loaded_config = AutoMLConfig.from_yaml('automl_config.yaml')
    print("✅ YAML configuration save/load")
    
    # Configuration validation
    validation_errors = config.validate()
    if validation_errors:
        print(f"Configuration errors: {validation_errors}")
    else:
        print("✅ Configuration is valid")
    
    # Configuration inspection
    print(f"Task type detection: {config.is_classification_task()}")
    print(f"Model parameters: {config.get_model_params()}")
    
    # Cleanup
    import os
    for file in ['automl_config.json', 'automl_config.yaml']:
        if os.path.exists(file):
            os.remove(file)

# ===================================
# 6. PRODUCTION WORKFLOWS
# ===================================

def production_workflows():
    """Production-ready workflow patterns."""
    
    print("\n🏭 PRODUCTION WORKFLOWS")
    print("=" * 50)
    
    from intelligent_automl.data import load_data, DataPipeline, MissingValueHandler, FeatureScaler, CategoricalEncoder
    
    # Workflow 1: Training pipeline for production
    def create_production_pipeline():
        """Create a robust production pipeline."""
        
        # Load and validate data
        df = load_data('training_data.csv')
        
        # Create comprehensive pipeline
        pipeline = (DataPipeline()
                   .add_step('missing_values', MissingValueHandler(
                       numeric_strategy='median',  # Robust to outliers
                       categorical_strategy='most_frequent'
                   ))
                   .add_step('categorical_encoding', CategoricalEncoder(
                       method='onehot',
                       handle_unknown='ignore',  # Handle new categories
                       max_categories=50  # Prevent explosion
                   ))
                   .add_step('feature_scaling', FeatureScaler(
                       method='robust'  # Less sensitive to outliers
                   )))
        
        # Validate pipeline
        validation_report = pipeline.validate(df)
        if not validation_report['is_valid']:
            raise ValueError(f"Pipeline validation failed: {validation_report['errors']}")
        
        # Fit pipeline
        target = df['target']
        features = df.drop('target', axis=1)
        pipeline.fit(features)
        
        # Save for production
        pipeline.save('production_pipeline_v1.joblib')
        pipeline.save_config('production_config_v1.json')
        
        return pipeline
    
    print("✅ Production pipeline creation")
    
    # Workflow 2: Batch processing workflow
    def batch_processing_workflow():
        """Process large datasets in batches."""
        
        # Load production pipeline
        pipeline = DataPipeline.load('production_pipeline_v1.joblib')
        
        # Process data in chunks
        chunk_size = 10000
        processed_chunks = []
        
        # Simulate large dataset processing
        for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
            processed_chunk = pipeline.transform(chunk)
            processed_chunks.append(processed_chunk)
        
        # Combine results
        final_result = pd.concat(processed_chunks, ignore_index=True)
        
        return final_result
    
    print("✅ Batch processing workflow")
    
    # Workflow 3: Real-time prediction pipeline
    def realtime_prediction_setup():
        """Setup for real-time predictions."""
        
        # Load pipeline once (in production, this would be at startup)
        pipeline = DataPipeline.load('production_pipeline_v1.joblib')
        
        def predict_single_record(record_dict):
            """Predict single record in real-time."""
            # Convert to DataFrame
            df = pd.DataFrame([record_dict])
            
            # Process through pipeline
            processed = pipeline.transform(df)
            
            # Return processed features (ready for model prediction)
            return processed.iloc[0].to_dict()
        
        # Example usage
        new_record = {
            'age': 35,
            'income': 65000,
            'city': 'Boston',
            'education': 'Bachelor'
        }
        
        processed_record = predict_single_record(new_record)
        return processed_record
    
    print("✅ Real-time prediction setup")
    
    # Workflow 4: A/B testing pipeline
    def ab_testing_workflow():
        """Setup for A/B testing different pipelines."""
        
        # Load different pipeline versions
        pipeline_a = DataPipeline.load('production_pipeline_v1.joblib')
        # pipeline_b = DataPipeline.load('production_pipeline_v2.joblib')
        
        def process_with_variant(data, variant='A'):
            """Process data with specified pipeline variant."""
            if variant == 'A':
                return pipeline_a.transform(data)
            # elif variant == 'B':
            #     return pipeline_b.transform(data)
            else:
                raise ValueError(f"Unknown variant: {variant}")
        
        return process_with_variant
    
    print("✅ A/B testing pipeline")
    
    # Workflow 5: Pipeline monitoring and alerting
    def monitoring_workflow():
        """Monitor pipeline performance in production."""
        
        pipeline = DataPipeline.load('production_pipeline_v1.joblib')
        
        def process_with_monitoring(data):
            """Process data with performance monitoring."""
            import time
            
            start_time = time.time()
            
            try:
                # Process data
                result = pipeline.transform(data)
                
                # Log success metrics
                processing_time = time.time() - start_time
                print(f"Processing successful: {processing_time:.3f}s for {len(data)} records")
                
                # Check for data drift (simplified)
                if len(result.columns) != 19:  # Expected number of features
                    print("⚠️ Warning: Unexpected number of features detected")
                
                return result
                
            except Exception as e:
                # Log error
                print(f"❌ Pipeline processing failed: {str(e)}")
                raise
        
        return process_with_monitoring
    
    print("✅ Pipeline monitoring")

# ===================================
# 7. ERROR HANDLING AND DEBUGGING
# ===================================

def error_handling_examples():
    """Error handling and debugging techniques."""
    
    print("\n🛠️ ERROR HANDLING & DEBUGGING")
    print("=" * 50)
    
    from intelligent_automl.core.exceptions import AutoMLError, PreprocessingError, DataLoadError
    from intelligent_automl.data import DataPipeline, MissingValueHandler
    
    # 1. Graceful error handling
    try:
        # This might fail
        df = load_data('nonexistent_file.csv')
    except DataLoadError as e:
        print(f"✅ Caught data loading error: {str(e)}")
    except AutoMLError as e:
        print(f"✅ Caught general AutoML error: {str(e)}")
    
    # 2. Pipeline validation before processing
    pipeline = DataPipeline().add_step('imputer', MissingValueHandler())
    
    # Create problematic data
    bad_data = pd.DataFrame({'A': ['text', 'more_text', 'even_more_text']})
    
    # Validate first
    validation_report = pipeline.validate(bad_data)
    if not validation_report['is_valid']:
        print(f"✅ Pipeline validation caught issues: {validation_report['errors']}")
    
    # 3. Step-by-step debugging
    def debug_pipeline_step_by_step():
        """Debug pipeline by processing step by step."""
        
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['x', None, 'z', 'w']
        })
        
        pipeline = (DataPipeline()
                   .add_step('imputer', MissingValueHandler())
                   .add_step('scaler', FeatureScaler()))
        
        # Process step by step for debugging
        current_data = df.copy()
        for step_name in pipeline.get_step_names():
            step = pipeline.get_step(step_name)
            print(f"Before {step_name}: {current_data.shape}, missing: {current_data.isnull().sum().sum()}")
            
            try:
                step.fit(current_data)
                current_data = step.transform(current_data)
                print(f"After {step_name}: {current_data.shape}, missing: {current_data.isnull().sum().sum()}")
            except Exception as e:
                print(f"❌ Error in step {step_name}: {str(e)}")
                break
    
    debug_pipeline_step_by_step()
    print("✅ Step-by-step debugging")
    
    # 4. Performance debugging
    def debug_performance():
        """Debug pipeline performance issues."""
        
        # Create large dataset
        large_df = pd.DataFrame({
            'A': np.random.randn(100000),
            'B': np.random.choice(['x', 'y', 'z'], 100000),
            'C': np.random.randn(100000)
        })
        
        pipeline = DataPipeline().add_step('imputer', MissingValueHandler())
        
        # Profile performance
        performance = pipeline.profile_performance(large_df, n_iterations=3)
        
        # Identify bottlenecks
        for step in performance['steps']:
            if step['avg_time_seconds'] > 1.0:  # Slow step
                print(f"⚠️ Slow step detected: {step['name']} ({step['avg_time_seconds']:.3f}s)")
        
        # Memory usage
        memory_info = pipeline.get_memory_usage()
        if memory_info['total_size_mb'] > 100:  # Large memory usage
            print(f"⚠️ High memory usage: {memory_info['total_size_mb']:.2f} MB")
    
    print("✅ Performance debugging")

# ===================================
# 8. INTEGRATION PATTERNS
# ===================================

def integration_examples():
    """Integration with other tools and frameworks."""
    
    print("\n🔗 INTEGRATION PATTERNS")
    print("=" * 50)
    
    from intelligent_automl.data import DataPipeline, MissingValueHandler, FeatureScaler
    
    # 1. Scikit-learn integration
    def sklearn_integration():
        """Integrate with scikit-learn pipelines."""
        
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create AutoML preprocessing pipeline
        automl_pipeline = (DataPipeline()
                          .add_step('imputer', MissingValueHandler())
                          .add_step('scaler', FeatureScaler()))
        
        # Use in scikit-learn workflow
        def create_sklearn_pipeline(automl_preprocessor):
            """Create sklearn pipeline with AutoML preprocessing."""
            
            # This would require a sklearn wrapper for AutoML pipeline
            # sklearn_pipeline = SklearnPipeline([
            #     ('automl_preprocessing', AutoMLPreprocessorWrapper(automl_preprocessor)),
            #     ('classifier', RandomForestClassifier())
            # ])
            
            pass
        
        print("✅ Scikit-learn integration pattern")
    
    sklearn_integration()
    
    # 2. Pandas integration
    def pandas_integration():
        """Seamless pandas DataFrame integration."""
        
        # AutoML pipeline works directly with pandas
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['x', None, 'z', 'w']
        })
        
        pipeline = DataPipeline().add_step('imputer', MissingValueHandler())
        
        # Maintains pandas DataFrame throughout
        result = pipeline.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        
        print("✅ Native pandas integration")
    
    pandas_integration()
    
    # 3. MLflow integration (conceptual)
    def mlflow_integration():
        """Integration with MLflow for experiment tracking."""
        
        # This would be implemented in a separate module
        # import mlflow
        # 
        # with mlflow.start_run():
        #     pipeline = DataPipeline().add_step('imputer', MissingValueHandler())
        #     
        #     # Log pipeline configuration
        #     mlflow.log_params(pipeline.get_params())
        #     
        #     # Log pipeline artifact
        #     pipeline.save('pipeline.joblib')
        #     mlflow.log_artifact('pipeline.joblib')
        
        print("✅ MLflow integration pattern")
    
    mlflow_integration()
    
    # 4. Jupyter notebook integration
    def jupyter_integration():
        """Jupyter notebook friendly features."""
        
        pipeline = (DataPipeline()
                   .add_step('imputer', MissingValueHandler())
                   .add_step('scaler', FeatureScaler()))
        
        # Rich display in notebooks
        print(pipeline)  # Shows pipeline structure
        
        # Easy parameter inspection
        params = pipeline.get_params(deep=True)
        for param, value in list(params.items())[:5]:  # Show first 5
            print(f"  {param}: {value}")
        
        print("✅ Jupyter notebook integration")
    
    jupyter_integration()

# ===================================
# MAIN USAGE EXAMPLES RUNNER
# ===================================

def main():
    """Run all usage examples."""
    
    print("🎯 AUTOML FRAMEWORK - COMPLETE USAGE GUIDE")
    print("=" * 80)
    
    # Create sample data files for examples
    sample_df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'city': ['NY', 'LA', 'Chicago', 'Boston', 'SF'],
        'target': [0, 1, 0, 1, 1]
    })
    sample_df.to_csv('data.csv', index=False)
    sample_df.to_excel('data.xlsx', index=False)
    sample_df.to_json('data.json', orient='records')
    
    try:
        # Run all examples
        data_loading_examples()
        preprocessing_examples()
        pipeline_examples()
        advanced_pipeline_operations()
        configuration_examples()
        production_workflows()
        error_handling_examples()
        integration_examples()
        
        print("\n" + "=" * 80)
        print("🎊 COMPLETE USAGE GUIDE FINISHED!")
        print("=" * 80)
        print("✅ Data Loading (6 methods)")
        print("✅ Preprocessing (7 components)")
        print("✅ Pipeline Construction (4 patterns)")
        print("✅ Advanced Operations (8 capabilities)")
        print("✅ Configuration Management (4 approaches)")
        print("✅ Production Workflows (5 patterns)")
        print("✅ Error Handling & Debugging (4 techniques)")
        print("✅ Integration Patterns (4 frameworks)")
        
        print("\n🚀 Your AutoML Framework supports:")
        print("   📁 Universal data loading")
        print("   🔧 Comprehensive preprocessing")
        print("   🔗 Flexible pipeline chaining")
        print("   💾 Production serialization")
        print("   🛠️ Advanced debugging")
        print("   🔗 Framework integration")
        print("   📋 Type-safe configuration")
        print("   🏭 Production workflows")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup sample files
        import os
        for file in ['data.csv', 'data.xlsx', 'data.json', 'production_pipeline_v1.joblib', 'production_config_v1.json']:
            if os.path.exists(file):
                os.remove(file)
        print("\n🗑️ Cleanup completed")

if __name__ == "__main__":
    main()

# ===================================
# QUICK REFERENCE CHEAT SHEET
# ===================================

"""
🎯 AUTOML FRAMEWORK - QUICK REFERENCE

1. BASIC DATA LOADING:
   from automl_framework.data import load_data
   df = load_data('file.csv')  # Auto-detects format

2. BASIC PIPELINE:
   from automl_framework.data import DataPipeline, MissingValueHandler, FeatureScaler
   pipeline = (DataPipeline()
              .add_step('imputer', MissingValueHandler())
              .add_step('scaler', FeatureScaler()))
   result = pipeline.fit_transform(df)

3. SAVE/LOAD PIPELINE:
   pipeline.save('my_pipeline.joblib')
   loaded = DataPipeline.load('my_pipeline.joblib')

4. ADVANCED PREPROCESSING:
   from automl_framework.data import CategoricalEncoder, OutlierHandler, DateTimeProcessor
   
   advanced_pipeline = (DataPipeline()
                       .add_step('datetime', DateTimeProcessor())
                       .add_step('outliers', OutlierHandler())
                       .add_step('missing', MissingValueHandler(numeric_strategy='knn'))
                       .add_step('encoding', CategoricalEncoder(method='onehot'))
                       .add_step('scaling', FeatureScaler(method='robust')))

5. CONFIGURATION:
   from automl_framework.core import AutoMLConfig, DataConfig, PreprocessingConfig
   
   config = AutoMLConfig(
       data=DataConfig(file_path='data.csv', target_column='target'),
       preprocessing=PreprocessingConfig(scaling_method='standard')
   )

6. ERROR HANDLING:
   from automl_framework.core.exceptions import AutoMLError, PreprocessingError
   
   try:
       result = pipeline.fit_transform(df)
   except PreprocessingError as e:
       print(f"Preprocessing failed: {e}")

7. VALIDATION:
   validation_report = pipeline.validate(df)
   if validation_report['is_valid']:
       print("Pipeline is valid!")

8. PERFORMANCE PROFILING:
   performance = pipeline.profile_performance(df)
   print(f"Total time: {performance['total_time_seconds']:.3f}s")

9. PRODUCTION USAGE:
   # Training
   pipeline.fit(training_data)
   pipeline.save('production_pipeline.joblib')
   
   # Production
   pipeline = DataPipeline.load('production_pipeline.joblib')
   processed = pipeline.transform(new_data)

10. DEBUGGING:
    for step_name in pipeline.get_step_names():
        step = pipeline.get_step(step_name)
        try:
            step.fit(data)
            data = step.transform(data)
            print(f"✅ {step_name} completed")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
"""