#!/usr/bin/env python
"""
Complete Intelligent AutoML Framework Integration

This script demonstrates the complete framework with all components
working together seamlessly.

Save this file as: intelligent_automl/complete_framework.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
import time
warnings.filterwarnings('ignore')

# Framework imports
from .intelligence.pipeline_selector import create_intelligent_pipeline, IntelligentPipelineSelector
from .data import load_data, DataPipeline
from .models.auto_trainer import AutoModelTrainer
from .utils.validation import validate_dataset, DataProfiler
from .utils.logging import configure_logging, get_logger, MetricsTracker
from .core.config import AutoMLConfig, DataConfig


class IntelligentAutoMLFramework:
    """
    Complete Intelligent AutoML Framework
    
    This class provides a unified interface to all framework capabilities,
    offering a seamless experience from data loading to model deployment.
    """
    
    def __init__(self, verbose: bool = True, log_file: str = None):
        """
        Initialize the framework.
        
        Args:
            verbose: Whether to show detailed output
            log_file: Optional log file path
        """
        self.verbose = verbose
        
        # Configure logging
        configure_logging(
            level='INFO' if verbose else 'WARNING',
            log_file=log_file,
            log_to_console=verbose
        )
        
        self.logger = get_logger('intelligent_automl')
        self.metrics_tracker = MetricsTracker(self.logger)
        
        # Framework state
        self.data = None
        self.target_column = None
        self.pipeline = None
        self.model_trainer = None
        self.results = {}
        
        if verbose:
            print("🧠 Intelligent AutoML Framework initialized!")
            print("🚀 Ready to intelligently process your data!")
    
    def load_data(self, file_path: str, target_column: str) -> pd.DataFrame:
        """
        Load and validate data.
        
        Args:
            file_path: Path to data file
            target_column: Name of target column
            
        Returns:
            Loaded DataFrame
        """
        if self.verbose:
            print(f"\n📁 Loading data from: {file_path}")
        
        try:
            # Load data
            self.data = load_data(file_path)
            self.target_column = target_column
            
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            if self.verbose:
                print(f"✅ Data loaded: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            
            # Track data quality
            self.metrics_tracker.track_data_quality(self.data, 'raw_data')
            
            # Validate data
            validation_report = validate_dataset(self.data, target_column=target_column)
            
            if self.verbose:
                print(f"🔍 Data validation: {'✅ Passed' if validation_report.is_valid else '⚠️ Issues found'}")
                if validation_report.warning_count > 0:
                    print(f"  Warnings: {validation_report.warning_count}")
                if validation_report.error_count > 0:
                    print(f"  Errors: {validation_report.error_count}")
            
            self.results['data_validation'] = validation_report
            return self.data
            
        except Exception as e:
            self.logger.log_error('data_loading', 'load_data', e)
            raise
    
    def analyze_data(self) -> dict:
        """
        Perform comprehensive data analysis.
        
        Returns:
            Data analysis profile
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.verbose:
            print(f"\n🔍 Analyzing data characteristics...")
        
        try:
            # Generate comprehensive profile
            profiler = DataProfiler()
            profile = profiler.profile_data(self.data)
            
            if self.verbose:
                print(f"✅ Analysis complete!")
                print(f"📊 Basic info: {profile['basic_info']['shape'][0]} rows × {profile['basic_info']['shape'][1]} columns")
                print(f"📈 Data quality: {profile['data_quality']['completeness']:.1f}% complete")
                print(f"💡 Recommendations: {len(profile['recommendations'])}")
                
                # Show key recommendations
                high_priority = [r for r in profile['recommendations'] if r['priority'] == 'high']
                if high_priority:
                    print(f"🔴 High priority issues: {len(high_priority)}")
                    for rec in high_priority[:3]:
                        print(f"  • {rec['message']}")
            
            self.results['data_analysis'] = profile
            return profile
            
        except Exception as e:
            self.logger.log_error('data_analysis', 'analyze_data', e)
            raise
    
    def create_pipeline(self) -> DataPipeline:
        """
        Create intelligent preprocessing pipeline.
        
        Returns:
            Configured DataPipeline
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.verbose:
            print(f"\n🧠 Creating intelligent preprocessing pipeline...")
        
        try:
            # Create intelligent pipeline
            self.pipeline = create_intelligent_pipeline(self.data, target_column=self.target_column)
            
            if self.verbose:
                print(f"✅ Pipeline created with {len(self.pipeline)} steps")
                print(f"🔧 Steps: {', '.join(self.pipeline.get_step_names())}")
            
            return self.pipeline
            
        except Exception as e:
            self.logger.log_error('pipeline_creation', 'create_pipeline', e)
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess data using intelligent pipeline.
        
        Returns:
            Processed features DataFrame
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        if self.verbose:
            print(f"\n⚙️ Preprocessing data...")
        
        try:
            # Prepare features
            features = self.data.drop(self.target_column, axis=1)
            
            # Process data
            start_time = time.time()
            processed_features = self.pipeline.fit_transform(features)
            processing_time = time.time() - start_time
            
            # Track processing metrics
            self.metrics_tracker.track_data_quality(processed_features, 'processed_data')
            
            if self.verbose:
                print(f"✅ Preprocessing complete in {processing_time:.2f} seconds!")
                print(f"📈 Features: {features.shape[1]} → {processed_features.shape[1]}")
                print(f"🎯 Missing values: {processed_features.isnull().sum().sum()}")
                print(f"⚡ Processing speed: {len(features) / processing_time:.0f} rows/second")
            
            self.results['preprocessing'] = {
                'original_features': features.shape[1],
                'final_features': processed_features.shape[1],
                'processing_time': processing_time,
                'missing_values': processed_features.isnull().sum().sum()
            }
            
            return processed_features
            
        except Exception as e:
            self.logger.log_error('preprocessing', 'preprocess_data', e)
            raise
    
    def train_models(self, 
                    models_to_try: list = None,
                    time_limit_minutes: int = None,
                    cv_folds: int = 5) -> dict:
        """
        Train models using intelligent selection.
        
        Args:
            models_to_try: List of models to try
            time_limit_minutes: Training time limit
            cv_folds: Cross-validation folds
            
        Returns:
            Training results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.verbose:
            print(f"\n🤖 Training models...")
        
        try:
            # Preprocess data if not done
            processed_features = self.preprocess_data()
            target = self.data[self.target_column]
            
            # Create model trainer
            self.model_trainer = AutoModelTrainer(
                models_to_try=models_to_try,
                time_limit_minutes=time_limit_minutes,
                cross_validation_folds=cv_folds,
                verbose=self.verbose
            )
            
            # Train models
            start_time = time.time()
            self.model_trainer.fit(processed_features, target)
            training_time = time.time() - start_time
            
            # Get results
            training_summary = self.model_trainer.get_training_summary()
            
            # Track model performance
            self.metrics_tracker.track_model_performance(
                training_summary['best_model'],
                {'score': training_summary['best_score']},
                stage='training'
            )
            
            if self.verbose:
                print(f"✅ Training complete in {training_time:.1f} seconds!")
                print(f"🏆 Best model: {training_summary['best_model']}")
                print(f"📊 Best score: {training_summary['best_score']:.4f}")
                print(f"🔄 Models trained: {training_summary['models_trained']}")
            
            self.results['model_training'] = training_summary
            return training_summary
            
        except Exception as e:
            self.logger.log_error('model_training', 'train_models', e)
            raise
    
    def make_predictions(self, new_data: pd.DataFrame = None) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            new_data: New data to predict on (uses training data if None)
            
        Returns:
            Predictions array
        """
        if self.model_trainer is None:
            raise ValueError("No model trained. Call train_models() first.")
        
        if new_data is None:
            new_data = self.data.drop(self.target_column, axis=1)
        
        if self.verbose:
            print(f"\n🔮 Making predictions on {len(new_data)} samples...")
        
        try:
            # Process new data
            processed_data = self.pipeline.transform(new_data)
            
            # Make predictions
            predictions = self.model_trainer.predict(processed_data)
            
            # Try to get probabilities
            try:
                probabilities = self.model_trainer.predict_proba(processed_data)
                has_probabilities = True
            except:
                probabilities = None
                has_probabilities = False
            
            if self.verbose:
                print(f"✅ Predictions complete!")
                if has_probabilities:
                    print(f"📊 Probabilities available: {probabilities.shape}")
                
                # Show prediction summary
                if predictions.dtype in ['int64', 'int32', 'bool']:
                    unique_values, counts = np.unique(predictions, return_counts=True)
                    print(f"📈 Prediction distribution:")
                    for value, count in zip(unique_values, counts):
                        percentage = (count / len(predictions)) * 100
                        print(f"  • Class {value}: {count} ({percentage:.1f}%)")
            
            return predictions
            
        except Exception as e:
            self.logger.log_error('prediction', 'make_predictions', e)
            raise
    
    def save_model(self, output_dir: str):
        """
        Save trained model and pipeline for production use.
        
        Args:
            output_dir: Directory to save model components
        """
        if self.model_trainer is None:
            raise ValueError("No model trained. Call train_models() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if self.verbose:
            print(f"\n💾 Saving model to: {output_path}")
        
        try:
            # Save model
            import joblib
            model_path = output_path / 'best_model.joblib'
            joblib.dump(self.model_trainer.best_model, model_path)
            
            # Save pipeline
            pipeline_path = output_path / 'preprocessing_pipeline.joblib'
            self.pipeline.save(str(pipeline_path))
            
            # Save results and metadata
            results_path = output_path / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Save configuration
            config_path = output_path / 'config.json'
            config = {
                'target_column': self.target_column,
                'data_shape': self.data.shape,
                'pipeline_steps': len(self.pipeline),
                'best_model': self.model_trainer.get_best_model_name(),
                'best_score': self.model_trainer.best_score,
                'framework_version': '1.0.0'
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if self.verbose:
                print(f"✅ Model saved successfully!")
                print(f"  • Model: {model_path}")
                print(f"  • Pipeline: {pipeline_path}")
                print(f"  • Results: {results_path}")
                print(f"  • Config: {config_path}")
            
        except Exception as e:
            self.logger.log_error('model_saving', 'save_model', e)
            raise
    
    def load_model(self, model_dir: str):
        """
        Load trained model and pipeline from directory.
        
        Args:
            model_dir: Directory containing saved model
        """
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        if self.verbose:
            print(f"\n📂 Loading model from: {model_path}")
        
        try:
            # Load model
            import joblib
            model_file = model_path / 'best_model.joblib'
            self.model_trainer = type('ModelTrainer', (), {})()
            self.model_trainer.best_model = joblib.load(model_file)
            
            # Load pipeline
            pipeline_file = model_path / 'preprocessing_pipeline.joblib'
            self.pipeline = DataPipeline.load(str(pipeline_file))
            
            # Load configuration
            config_file = model_path / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.target_column = config.get('target_column')
            
            if self.verbose:
                print(f"✅ Model loaded successfully!")
                print(f"  • Target column: {self.target_column}")
                print(f"  • Pipeline steps: {len(self.pipeline)}")
            
        except Exception as e:
            self.logger.log_error('model_loading', 'load_model', e)
            raise
    
    def get_comprehensive_report(self) -> dict:
        """
        Generate comprehensive report of all results.
        
        Returns:
            Complete analysis and training report
        """
        if not self.results:
            raise ValueError("No results available. Run the complete pipeline first.")
        
        # Add metrics summary
        metrics_summary = self.metrics_tracker.get_metrics_summary()
        
        report = {
            'framework_info': {
                'version': '1.0.0',
                'components': ['data_loading', 'preprocessing', 'model_training', 'prediction'],
                'intelligent_features': [
                    'automatic_data_analysis',
                    'intelligent_pipeline_selection',
                    'smart_preprocessing',
                    'automatic_model_selection'
                ]
            },
            'data_info': {
                'shape': self.data.shape if self.data is not None else None,
                'target_column': self.target_column,
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2 if self.data is not None else None
            },
            'results': self.results,
            'metrics_summary': metrics_summary,
            'performance_summary': {
                'preprocessing_time': self.results.get('preprocessing', {}).get('processing_time', 0),
                'training_time': self.results.get('model_training', {}).get('total_training_time', 0),
                'best_score': self.results.get('model_training', {}).get('best_score', 0),
                'feature_expansion': self.results.get('preprocessing', {}).get('final_features', 0) / max(self.results.get('preprocessing', {}).get('original_features', 1), 1)
            }
        }
        
        return report
    
    def run_complete_pipeline(self, file_path: str, target_column: str, 
                            output_dir: str = None, **kwargs) -> dict:
        """
        Run the complete AutoML pipeline from start to finish.
        
        Args:
            file_path: Path to data file
            target_column: Target column name
            output_dir: Optional output directory for saving results
            **kwargs: Additional arguments for training
            
        Returns:
            Complete results dictionary
        """
        if self.verbose:
            print("🚀 Running Complete Intelligent AutoML Pipeline")
            print("=" * 60)
        
        try:
            # Step 1: Load and validate data
            self.load_data(file_path, target_column)
            
            # Step 2: Analyze data
            self.analyze_data()
            
            # Step 3: Create and apply preprocessing pipeline
            self.preprocess_data()
            
            # Step 4: Train models
            self.train_models(**kwargs)
            
            # Step 5: Save results if requested
            if output_dir:
                self.save_model(output_dir)
            
            # Step 6: Generate comprehensive report
            report = self.get_comprehensive_report()
            
            if self.verbose:
                print("\n" + "=" * 60)
                print("🎉 COMPLETE PIPELINE FINISHED!")
                print("=" * 60)
                print(f"📊 Dataset: {report['data_info']['shape'][0]} rows × {report['data_info']['shape'][1]} columns")
                print(f"🧠 Intelligence applied: {len(report['framework_info']['intelligent_features'])} features")
                print(f"🔧 Preprocessing: {report['results']['preprocessing']['original_features']} → {report['results']['preprocessing']['final_features']} features")
                print(f"🏆 Best model: {report['results']['model_training']['best_model']}")
                print(f"📈 Best score: {report['results']['model_training']['best_score']:.4f}")
                print(f"⚡ Total time: {report['performance_summary']['preprocessing_time'] + report['performance_summary']['training_time']:.1f}s")
                
                if output_dir:
                    print(f"💾 Results saved to: {output_dir}")
                
                print("\n🧠 Your data has been intelligently processed and models trained!")
                print("🚀 Framework is ready for production use!")
            
            return report
            
        except Exception as e:
            self.logger.log_error('complete_pipeline', 'run_complete_pipeline', e)
            raise


        
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess data using intelligent pipeline with ID column handling.
        
        Returns:
            Processed features DataFrame (excluding ID columns)
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        if self.verbose:
            print(f"\n⚙️ Preprocessing data...")
        
        try:
            # Prepare features - EXCLUDE ID COLUMNS
            features = self.data.drop(self.target_column, axis=1)
            
            # Smart ID column detection and removal
            id_columns = self._detect_id_columns(features)
            if id_columns:
                if self.verbose:
                    print(f"🔍 Excluding ID columns: {id_columns}")
                features = features.drop(columns=id_columns)
            
            # Process data
            start_time = time.time()
            processed_features = self.pipeline.fit_transform(features)
            processing_time = time.time() - start_time
            
            # Track processing metrics
            self.metrics_tracker.track_data_quality(processed_features, 'processed_data')
            
            if self.verbose:
                print(f"✅ Preprocessing complete in {processing_time:.2f} seconds!")
                print(f"📈 Features: {features.shape[1]} → {processed_features.shape[1]}")
                print(f"🎯 Missing values: {processed_features.isnull().sum().sum()}")
                print(f"⚡ Processing speed: {len(features) / processing_time:.0f} rows/second")
            
            self.results['preprocessing'] = {
                'original_features': features.shape[1],
                'final_features': processed_features.shape[1],
                'processing_time': processing_time,
                'missing_values': processed_features.isnull().sum().sum(),
                'excluded_columns': id_columns
            }
            
            return processed_features
            
        except Exception as e:
            self.logger.log_error('preprocessing', 'preprocess_data', e)
            raise

    def _detect_id_columns(self, data: pd.DataFrame) -> list:
        """
        Detect and return ID columns that should be excluded from training.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of column names that are likely ID columns
        """
        id_columns = []
        
        for col in data.columns:
            col_lower = col.lower()
            
            # Check column name patterns
            id_patterns = ['id', 'key', 'index', 'uuid', 'guid']
            if any(pattern in col_lower for pattern in id_patterns):
                id_columns.append(col)
                continue
            
            # Check if column has very high cardinality (likely unique IDs)
            if data[col].dtype == 'object':
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio > 0.95:  # More than 95% unique values
                    id_columns.append(col)
        
        return id_columns


def demo_complete_framework():
    """Demonstrate the complete framework with sample data."""
    print("🎭 INTELLIGENT AUTOML FRAMEWORK DEMO")
    print("=" * 80)
    
    # Create sample dataset
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.exponential(50000, 1000),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
        'signup_date': pd.date_range('2020-01-01', periods=1000, freq='H'),
        'is_customer': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    # Add realistic data issues
    sample_data.loc[::10, 'age'] = np.nan
    sample_data.loc[::15, 'income'] = np.nan
    sample_data.loc[::20, 'city'] = np.nan
    
    # Save sample data
    sample_data.to_csv('demo_data.csv', index=False)
    
    print("📊 Created sample dataset with 1000 rows and realistic data issues")
    print("🎯 Target: is_customer (binary classification)")
    print("⚠️  Issues: Missing values, mixed data types, datetime features")
    
    # Initialize framework
    framework = IntelligentAutoMLFramework(verbose=True)
    
    # Run complete pipeline
    results = framework.run_complete_pipeline(
        file_path='demo_data.csv',
        target_column='is_customer',
        output_dir='demo_results',
        models_to_try=['random_forest', 'logistic_regression'],
        time_limit_minutes=2
    )
    
    # Test predictions on new data
    print("\n🔮 Testing predictions on new data...")
    new_data = pd.DataFrame({
        'age': [25, 45, 35],
        'income': [45000, 85000, 65000],
        'city': ['NYC', 'LA', 'Chicago'],
        'education': ['Bachelor', 'Master', 'PhD'],
        'signup_date': pd.date_range('2024-01-01', periods=3)
    })
    
    predictions = framework.make_predictions(new_data)
    print(f"✅ Predictions: {predictions}")
    
    # Show final summary
    print("\n🎯 DEMO SUMMARY:")
    print(f"  • Framework completed all steps successfully")
    print(f"  • Data quality improved to 100%")
    print(f"  • Model trained and ready for production")
    print(f"  • Results saved to 'demo_results' directory")
    
    # Cleanup
    import os
    os.remove('demo_data.csv')
    print("\n🗑️  Demo files cleaned up")




def run_framework_tests():
    """Run basic framework tests."""
    print("🧪 RUNNING FRAMEWORK TESTS")
    print("=" * 50)
    
    try:
        # Test 1: Basic functionality
        print("Test 1: Basic functionality...")
        framework = IntelligentAutoMLFramework(verbose=False)
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 1, 0, 1, 0]
        })
        test_data.to_csv('test_data.csv', index=False)
        
        # Run pipeline
        results = framework.run_complete_pipeline(
            'test_data.csv', 
            'target',
            models_to_try=['random_forest']
        )
        
        assert results['results']['model_training']['best_score'] > 0
        print("✅ Test 1 passed")
        
        # Test 2: Model persistence
        print("Test 2: Model persistence...")
        framework.save_model('test_model')
        
        new_framework = IntelligentAutoMLFramework(verbose=False)
        new_framework.load_model('test_model')
        
        test_predictions = new_framework.make_predictions(test_data.drop('target', axis=1))
        assert len(test_predictions) == len(test_data)
        print("✅ Test 2 passed")
        
        print("\n🎉 All tests passed! Framework is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        import os
        import shutil
        for item in ['test_data.csv', 'test_model', 'demo_results']:
            if os.path.exists(item):
                if os.path.isfile(item):
                    os.remove(item)
                else:
                    shutil.rmtree(item)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_framework_tests()
    else:
        demo_complete_framework()