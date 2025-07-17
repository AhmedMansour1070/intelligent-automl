#!/usr/bin/env python
"""
Comprehensive test suite for the Intelligent AutoML Framework.

This module provides thorough testing of all framework components
including unit tests, integration tests, and end-to-end scenarios.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all framework components
from intelligent_automl import create_intelligent_pipeline, IntelligentPipelineSelector
from intelligent_automl.data import (
    load_data, DataPipeline, MissingValueHandler, FeatureScaler, 
    CategoricalEncoder, OutlierHandler, DateTimeProcessor
)
from intelligent_automl.models.auto_trainer import AutoModelTrainer
from intelligent_automl.utils.validation import validate_dataset, DataProfiler
from intelligent_automl.utils.logging import get_logger, MetricsTracker
from intelligent_automl.core.exceptions import AutoMLError, PreprocessingError
from intelligent_automl.core.config import AutoMLConfig, DataConfig


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_load_csv(self):
        """Test CSV loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            
            # Test loading
            loaded_data = load_data(f.name)
            self.assertEqual(loaded_data.shape, self.test_data.shape)
            self.assertListEqual(list(loaded_data.columns), list(self.test_data.columns))
            
            # Cleanup
            os.unlink(f.name)
    
    def test_load_excel(self):
        """Test Excel loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            self.test_data.to_excel(f.name, index=False)
            
            # Test loading
            loaded_data = load_data(f.name)
            self.assertEqual(loaded_data.shape, self.test_data.shape)
            
            # Cleanup
            os.unlink(f.name)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with self.assertRaises(Exception):
            load_data('nonexistent_file.csv')


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing components."""
    
    def setUp(self):
        """Set up test data with various issues."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'numeric_with_missing': [1, 2, np.nan, 4, 5],
            'categorical_with_missing': ['A', 'B', None, 'A', 'B'],
            'skewed_numeric': [1, 2, 3, 100, 200],
            'datetime_col': pd.date_range('2020-01-01', periods=5),
            'constant_col': [1, 1, 1, 1, 1],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_missing_value_handler(self):
        """Test missing value handling."""
        handler = MissingValueHandler()
        result = handler.fit_transform(self.test_data)
        
        # Should have no missing values
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Should preserve shape
        self.assertEqual(result.shape, self.test_data.shape)
    
    def test_feature_scaler(self):
        """Test feature scaling."""
        scaler = FeatureScaler(method='standard')
        result = scaler.fit_transform(self.test_data)
        
        # Numeric columns should be scaled
        numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in result.columns:
                # Check if values are approximately standardized (mean~0, std~1)
                col_mean = result[col].mean()
                col_std = result[col].std()
                self.assertAlmostEqual(col_mean, 0, places=1)
                self.assertAlmostEqual(col_std, 1, places=1)
    
    def test_categorical_encoder(self):
        """Test categorical encoding."""
        encoder = CategoricalEncoder(method='onehot')
        result = encoder.fit_transform(self.test_data)
        
        # Should have more columns due to one-hot encoding
        self.assertGreater(result.shape[1], self.test_data.shape[1])
        
        # Should have no missing values after encoding
        self.assertEqual(result.isnull().sum().sum(), 0)
    
    def test_datetime_processor(self):
        """Test datetime processing."""
        processor = DateTimeProcessor()
        result = processor.fit_transform(self.test_data)
        
        # Should have more columns due to datetime feature extraction
        self.assertGreater(result.shape[1], self.test_data.shape[1])
        
        # Should have datetime-related columns
        datetime_features = [col for col in result.columns if 'datetime_col' in col]
        self.assertGreater(len(datetime_features), 0)
    
    def test_outlier_handler(self):
        """Test outlier handling."""
        handler = OutlierHandler(method='iqr', treatment='cap')
        result = handler.fit_transform(self.test_data)
        
        # Should preserve shape
        self.assertEqual(result.shape, self.test_data.shape)
        
        # Extreme values should be capped
        skewed_col = 'skewed_numeric'
        if skewed_col in result.columns:
            self.assertLess(result[skewed_col].max(), self.test_data[skewed_col].max())


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': ['A', 'B', None, 'A', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_pipeline_creation(self):
        """Test pipeline creation and chaining."""
        pipeline = (DataPipeline()
                   .add_step('missing', MissingValueHandler())
                   .add_step('encoding', CategoricalEncoder())
                   .add_step('scaling', FeatureScaler()))
        
        # Check pipeline structure
        self.assertEqual(len(pipeline), 3)
        self.assertListEqual(pipeline.get_step_names(), ['missing', 'encoding', 'scaling'])
    
    def test_pipeline_execution(self):
        """Test pipeline execution."""
        pipeline = (DataPipeline()
                   .add_step('missing', MissingValueHandler())
                   .add_step('encoding', CategoricalEncoder()))
        
        result = pipeline.fit_transform(self.test_data)
        
        # Should have no missing values
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Should have more columns due to encoding
        self.assertGreater(result.shape[1], self.test_data.shape[1])
    
    def test_pipeline_serialization(self):
        """Test pipeline save/load."""
        pipeline = (DataPipeline()
                   .add_step('missing', MissingValueHandler())
                   .add_step('scaling', FeatureScaler()))
        
        # Fit pipeline
        pipeline.fit(self.test_data)
        
        # Save pipeline
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            pipeline.save(f.name)
            
            # Load pipeline
            loaded_pipeline = DataPipeline.load(f.name)
            
            # Test that loaded pipeline works
            result = loaded_pipeline.transform(self.test_data)
            self.assertEqual(result.shape[0], self.test_data.shape[0])
            
            # Cleanup
            os.unlink(f.name)
    
    def test_pipeline_validation(self):
        """Test pipeline validation."""
        pipeline = (DataPipeline()
                   .add_step('missing', MissingValueHandler())
                   .add_step('scaling', FeatureScaler()))
        
        validation_report = pipeline.validate(self.test_data)
        
        # Should return validation report
        self.assertIsInstance(validation_report, dict)
        self.assertIn('is_valid', validation_report)
        self.assertIn('errors', validation_report)


class TestIntelligentPipelineSelector(unittest.TestCase):
    """Test intelligent pipeline selection."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'age': np.random.normal(35, 10, 100),
            'income': np.random.exponential(50000, 100),
            'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100),
            'signup_date': pd.date_range('2020-01-01', periods=100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Add missing values
        self.test_data.loc[::10, 'age'] = np.nan
        self.test_data.loc[::15, 'city'] = np.nan
    
    def test_data_analysis(self):
        """Test data analysis functionality."""
        selector = IntelligentPipelineSelector(target_column='target')
        characteristics = selector.analyze_data(self.test_data)
        
        # Should detect data characteristics
        self.assertGreater(len(characteristics.numeric_features), 0)
        self.assertGreater(len(characteristics.categorical_features), 0)
        self.assertGreater(len(characteristics.datetime_features), 0)
        self.assertGreater(characteristics.missing_percentage, 0)
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        selector = IntelligentPipelineSelector(target_column='target')
        selector.analyze_data(self.test_data)
        recommendations = selector.generate_recommendations()
        
        # Should generate recommendations
        self.assertGreater(len(recommendations), 0)
        
        # Each recommendation should have required fields
        for rec in recommendations:
            self.assertIsInstance(rec.step_name, str)
            self.assertIsInstance(rec.confidence, float)
            self.assertTrue(0 <= rec.confidence <= 1)
            self.assertIsInstance(rec.priority, int)
    
    def test_pipeline_building(self):
        """Test intelligent pipeline building."""
        selector = IntelligentPipelineSelector(target_column='target')
        selector.analyze_data(self.test_data)
        pipeline = selector.build_intelligent_pipeline()
        
        # Should create a valid pipeline
        self.assertIsInstance(pipeline, DataPipeline)
        self.assertGreater(len(pipeline), 0)
        
        # Pipeline should process data successfully
        features = self.test_data.drop('target', axis=1)
        result = pipeline.fit_transform(features)
        
        # Should have no missing values
        self.assertEqual(result.isnull().sum().sum(), 0)
    
    def test_create_intelligent_pipeline_convenience(self):
        """Test convenience function."""
        pipeline = create_intelligent_pipeline(self.test_data, target_column='target')
        
        # Should create a valid pipeline
        self.assertIsInstance(pipeline, DataPipeline)
        
        # Should process data successfully
        features = self.test_data.drop('target', axis=1)
        result = pipeline.fit_transform(features)
        
        # Should improve data quality
        self.assertEqual(result.isnull().sum().sum(), 0)
        self.assertGreaterEqual(result.shape[1], features.shape[1])


class TestAutoModelTrainer(unittest.TestCase):
    """Test automatic model training."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Classification data
        self.classification_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.normal(0, 1, 200),
            'feature3': np.random.normal(0, 1, 200),
            'target': np.random.choice([0, 1], 200)
        })
        
        # Regression data
        self.regression_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.normal(0, 1, 200),
            'feature3': np.random.normal(0, 1, 200),
            'target': np.random.normal(50, 10, 200)
        })
    
    def test_classification_training(self):
        """Test classification model training."""
        trainer = AutoModelTrainer(
            task_type='classification',
            models_to_try=['random_forest', 'logistic_regression'],
            verbose=False
        )
        
        X = self.classification_data.drop('target', axis=1)
        y = self.classification_data['target']
        
        trainer.fit(X, y)
        
        # Should have trained models
        self.assertGreater(len(trainer.trained_models), 0)
        self.assertIsNotNone(trainer.best_model)
        
        # Should be able to make predictions
        predictions = trainer.predict(X)
        self.assertEqual(len(predictions), len(y))
    
    def test_auto_task_detection(self):
        """Test automatic task type detection."""
        # Classification task detection
        trainer_cls = AutoModelTrainer(verbose=False)
        X_cls = self.classification_data.drop('target', axis=1)
        y_cls = self.classification_data['target']
        
        detected_task = trainer_cls._detect_task_type(y_cls)
        self.assertEqual(detected_task, 'classification')
        
        # Regression task detection
        trainer_reg = AutoModelTrainer(verbose=False)
        X_reg = self.regression_data.drop('target', axis=1)
        y_reg = self.regression_data['target']
        
        detected_task = trainer_reg._detect_task_type(y_reg)
        self.assertEqual(detected_task, 'regression')
    
    def test_model_rankings(self):
        """Test model performance rankings."""
        trainer = AutoModelTrainer(
            models_to_try=['random_forest', 'logistic_regression'],
            verbose=False
        )
        
        X = self.classification_data.drop('target', axis=1)
        y = self.classification_data['target']
        
        trainer.fit(X, y)
        
        rankings = trainer.get_model_rankings()
        
        # Should have rankings for all models
        self.assertEqual(len(rankings), len(trainer.trained_models))
        
        # Rankings should be sorted by score
        scores = [rank['score'] for rank in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_training_summary(self):
        """Test training summary generation."""
        trainer = AutoModelTrainer(verbose=False)
        
        X = self.classification_data.drop('target', axis=1)
        y = self.classification_data['target']
        
        trainer.fit(X, y)
        
        summary = trainer.get_training_summary()
        
        # Should contain required fields
        required_fields = ['best_model', 'best_score', 'models_trained', 'total_training_time']
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Should have trained models
        self.assertGreater(len(trainer.trained_models), 0)
        self.assertIsNotNone(trainer.best_model)
        
        # Should be able to make predictions
        predictions = trainer.predict(X)
        self.assertEqual(len(predictions), len(y))
    
    def test_auto_task_detection(self):
        """Test automatic task type detection."""
        # Classification task detection
        trainer_cls = AutoModelTrainer(verbose=False)
        X_cls = self.classification_data.drop('target', axis=1)
        y_cls = self.classification_data['target']
        
        detected_task = trainer_cls._detect_task_type(y_cls)
        self.assertEqual(detected_task, 'classification')
        
        # Regression task detection
        trainer_reg = AutoModelTrainer(verbose=False)
        X_reg = self.regression_data.drop('target', axis=1)
        y_reg = self.regression_data['target']
        
        detected_task = trainer_reg._detect_task_type(y_reg)
        self.assertEqual(detected_task, 'regression')
    
    def test_model_rankings(self):
        """Test model performance rankings."""
        trainer = AutoModelTrainer(
            models_to_try=['random_forest', 'logistic_regression'],
            verbose=False
        )
        
        X = self.classification_data.drop('target', axis=1)
        y = self.classification_data['target']
        
        trainer.fit(X, y)
        
        rankings = trainer.get_model_rankings()
        
        # Should have rankings for all models
        self.assertEqual(len(rankings), len(trainer.trained_models))
        
        # Rankings should be sorted by score
        scores = [rank['score'] for rank in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_training_summary(self):
        """Test training summary generation."""
        trainer = AutoModelTrainer(verbose=False)
        
        X = self.classification_data.drop('target', axis=1)
        y = self.classification_data['target']
        
        trainer.fit(X, y)
        
        summary = trainer.get_training_summary()
        
        # Should contain required fields
        required_fields = ['best_model', 'best_score', 'models_trained', 'total_training_time']
        for field in required_fields:
            self.assertIn(field, summary)


class TestValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.good_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 1, 0, 1, 0]
        })
        
        self.bad_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, np.nan, np.nan],  # Lots of missing
            'feature2': ['A', 'A', 'A', 'A', 'A'],      # No variance
            'target': [0, 0, 0, 0, 0]                   # No variance
        })
    
    def test_validate_good_data(self):
        """Test validation of good quality data."""
        report = validate_dataset(self.good_data, target_column='target')
        
        # Should have fewer issues
        self.assertLessEqual(report.error_count, 1)
        self.assertIsInstance(report.is_valid, bool)
        self.assertIsInstance(report.results, list)
    
    def test_validate_bad_data(self):
        """Test validation of poor quality data."""
        report = validate_dataset(self.bad_data, target_column='target')
        
        # Should detect issues
        self.assertGreater(len(report.results), 0)
        
        # Should have specific validation results
        for result in report.results:
            self.assertIsInstance(result.message, str)
            self.assertIn(result.severity, ['error', 'warning', 'info'])
    
    def test_data_profiler(self):
        """Test data profiling functionality."""
        profiler = DataProfiler()
        profile = profiler.profile_data(self.good_data)
        
        # Should contain required sections
        required_sections = ['basic_info', 'column_profiles', 'data_quality', 'recommendations']
        for section in required_sections:
            self.assertIn(section, profile)
        
        # Basic info should be correct
        self.assertEqual(profile['basic_info']['shape'], self.good_data.shape)
        self.assertEqual(profile['basic_info']['column_count'], len(self.good_data.columns))
        
        # Should have column profiles for each column
        self.assertEqual(len(profile['column_profiles']), len(self.good_data.columns))


class TestLogging(unittest.TestCase):
    """Test logging and monitoring functionality."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_logger('test_logger')
        
        # Should create valid logger
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_logger')
    
    def test_metrics_tracking(self):
        """Test metrics tracking."""
        tracker = MetricsTracker()
        
        # Track some metrics
        test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        tracker.track_data_quality(test_data, 'test_stage')
        
        tracker.track_model_performance(
            'test_model', 
            {'accuracy': 0.85, 'precision': 0.80}
        )
        
        # Should have tracked metrics
        summary = tracker.get_metrics_summary()
        self.assertGreater(summary['total_records'], 0)
        self.assertIn('data_quality_checks', summary)
        self.assertIn('model_evaluations', summary)


class TestExceptions(unittest.TestCase):
    """Test exception handling."""
    
    def test_automl_error(self):
        """Test AutoML error handling."""
        with self.assertRaises(AutoMLError):
            raise AutoMLError("Test error", details={'test': 'value'})
    
    def test_preprocessing_error(self):
        """Test preprocessing error handling."""
        with self.assertRaises(PreprocessingError):
            raise PreprocessingError("Test preprocessing error")
    
    def test_error_inheritance(self):
        """Test error inheritance."""
        # PreprocessingError should inherit from AutoMLError
        self.assertTrue(issubclass(PreprocessingError, AutoMLError))


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        data_config = DataConfig(
            file_path='test.csv',
            target_column='target'
        )
        
        config = AutoMLConfig(data=data_config)
        
        # Should create valid configuration
        self.assertIsNotNone(config)
        self.assertEqual(config.data.file_path, 'test.csv')
        self.assertEqual(config.data.target_column, 'target')
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Create invalid config (non-existent file)
        data_config = DataConfig(
            file_path='nonexistent.csv',
            target_column='target'
        )
        
        # Should raise configuration error
        with self.assertRaises(Exception):
            AutoMLConfig(data=data_config)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame({'a': [1, 2], 'target': [0, 1]}).to_csv(f.name, index=False)
            
            data_config = DataConfig(
                file_path=f.name,
                target_column='target'
            )
            
            config = AutoMLConfig(data=data_config)
            
            # Test JSON serialization
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
                config.to_json(json_file.name)
                
                # Load and compare
                loaded_config = AutoMLConfig.from_json(json_file.name)
                self.assertEqual(loaded_config.data.target_column, 'target')
                
                # Cleanup
                os.unlink(json_file.name)
            
            # Cleanup
            os.unlink(f.name)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create realistic dataset
        self.test_data = pd.DataFrame({
            'age': np.random.normal(35, 10, 500),
            'income': np.random.exponential(50000, 500),
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 500),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
            'signup_date': pd.date_range('2020-01-01', periods=500),
            'target': np.random.choice([0, 1], 500, p=[0.7, 0.3])
        })
        
        # Introduce realistic data issues
        self.test_data.loc[::10, 'age'] = np.nan
        self.test_data.loc[::15, 'income'] = np.nan
        self.test_data.loc[::20, 'city'] = np.nan
        
        # Add some duplicates
        self.test_data = pd.concat([self.test_data, self.test_data.iloc[:10]], ignore_index=True)
    
    def test_complete_automl_pipeline(self):
        """Test complete AutoML pipeline."""
        # Step 1: Data Analysis
        profiler = DataProfiler()
        profile = profiler.profile_data(self.test_data)
        
        # Should generate meaningful profile
        self.assertGreater(len(profile['recommendations']), 0)
        
        # Step 2: Intelligent Preprocessing
        pipeline = create_intelligent_pipeline(self.test_data, target_column='target')
        
        features = self.test_data.drop('target', axis=1)
        processed_features = pipeline.fit_transform(features)
        
        # Should improve data quality
        self.assertEqual(processed_features.isnull().sum().sum(), 0)
        self.assertGreaterEqual(processed_features.shape[1], features.shape[1])
        
        # Step 3: Model Training
        trainer = AutoModelTrainer(verbose=False)
        trainer.fit(processed_features, self.test_data['target'])
        
        # Should successfully train models
        self.assertGreater(len(trainer.trained_models), 0)
        self.assertIsNotNone(trainer.best_model)
        
        # Step 4: Predictions
        predictions = trainer.predict(processed_features)
        
        # Should generate valid predictions
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Step 5: Model Serialization
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as model_file:
            import joblib
            joblib.dump(trainer.best_model, model_file.name)
            
            # Load and test
            loaded_model = joblib.load(model_file.name)
            loaded_predictions = loaded_model.predict(processed_features)
            
            # Should get same predictions
            np.testing.assert_array_equal(predictions, loaded_predictions)
            
            # Cleanup
            os.unlink(model_file.name)
        
        # Step 6: Pipeline Serialization
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as pipeline_file:
            pipeline.save(pipeline_file.name)
            
            # Load and test
            loaded_pipeline = DataPipeline.load(pipeline_file.name)
            loaded_processed = loaded_pipeline.transform(features)
            
            # Should get same processed data
            pd.testing.assert_frame_equal(processed_features, loaded_processed)
            
            # Cleanup
            os.unlink(pipeline_file.name))
        
        # Should improve data quality
        self.assertEqual(processed_features.isnull().sum().sum(), 0)
        self.assertGreaterEqual(processed_features.shape[1], features.shape[1])
        
        # Step 3: Model Training
        trainer = AutoModelTrainer(verbose=False)
        trainer.fit(processed_features, self.test_data['target'])
        
        # Should successfully train models
        self.assertGreater(len(trainer.trained_models), 0)
        self.assertIsNotNone(trainer.best_model)
        
        # Step 4: Predictions
        predictions = trainer.predict(processed_features)
        
        # Should generate valid predictions
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Step 5: Model Serialization
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as model_file:
            import joblib
            joblib.dump(trainer.best_model, model_file.name)
            
            # Load and test
            loaded_model = joblib.load(model_file.name)
            loaded_predictions = loaded_model.predict(processed_features)
            
            # Should get same predictions
            np.testing.assert_array_equal(predictions, loaded_predictions)
            
            # Cleanup
            os.unlink(model_file.name)
        
        # Step 6: Pipeline Serialization
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as pipeline_file:
            pipeline.save(pipeline_file.name)
            
            # Load and test
            loaded_pipeline = DataPipeline.load(pipeline_file.name)
            loaded_processed = loaded_pipeline.transform(features)
            
            # Should get same processed data
            pd.testing.assert_frame_equal(processed_features, loaded_processed)
            
            # Cleanup
            os.unlink(pipeline_file.name)
    
    def test_automl_with_different_data_types(self):
        """Test AutoML with various data types and issues."""
        # Test with regression data
        regression_data = self.test_data.copy()
        regression_data['target'] = np.random.normal(100, 20, len(regression_data))
        
        # Should handle regression task
        pipeline = create_intelligent_pipeline(regression_data, target_column='target')
        features = regression_data.drop('target', axis=1)
        processed_features = pipeline.fit_transform(features)
        
        trainer = AutoModelTrainer(task_type='regression', verbose=False)
        trainer.fit(processed_features, regression_data['target'])
        
        # Should successfully train regression models
        self.assertGreater(len(trainer.trained_models), 0)
        self.assertIsNotNone(trainer.best_model)
        
        # Test with time series data
        time_series_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100).cumsum(),
            'category': np.random.choice(['A', 'B'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Should handle datetime features
        pipeline = create_intelligent_pipeline(time_series_data, target_column='target')
        features = time_series_data.drop('target', axis=1)
        processed_features = pipeline.fit_transform(features)
        
        # Should extract datetime features
        self.assertGreater(processed_features.shape[1], features.shape[1])
        
        # Test with high-cardinality categorical data
        high_card_data = pd.DataFrame({
            'id': [f'id_{i}' for i in range(200)],  # High cardinality
            'feature': np.random.randn(200),
            'target': np.random.choice([0, 1], 200)
        })
        
        # Should handle high cardinality appropriately
        pipeline = create_intelligent_pipeline(high_card_data, target_column='target')
        features = high_card_data.drop('target', axis=1)
        processed_features = pipeline.fit_transform(features)
        
        # Should not explode in dimensionality
        self.assertLess(processed_features.shape[1], 50)  # Reasonable limit


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'feature1': np.random.randn(10000),
            'feature2': np.random.randn(10000),
            'feature3': np.random.choice(['A', 'B', 'C'], 10000),
            'target': np.random.choice([0, 1], 10000)
        })
        
        # Add missing values
        large_data.loc[::100, 'feature1'] = np.nan
        
        # Should process reasonably quickly
        import time
        start_time = time.time()
        
        pipeline = create_intelligent_pipeline(large_data, target_column='target')
        features = large_data.drop('target', axis=1)
        processed_features = pipeline.fit_transform(features)
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 30)  # 30 seconds
        
        # Should maintain data quality
        self.assertEqual(processed_features.isnull().sum().sum(), 0)
        self.assertEqual(processed_features.shape[0], large_data.shape[0])
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        # Create dataset
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        # Process data
        pipeline = create_intelligent_pipeline(data, target_column='target')
        features = data.drop('target', axis=1)
        processed_features = pipeline.fit_transform(features)
        
        # Memory usage should be reasonable
        original_memory = data.memory_usage(deep=True).sum()
        processed_memory = processed_features.memory_usage(deep=True).sum()
        
        # Should not use excessive memory (less than 10x original)
        self.assertLess(processed_memory, original_memory * 10)


def run_all_tests():
    """Run all tests and generate report."""
    print("🧪 Running Comprehensive Test Suite for Intelligent AutoML Framework")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataLoading,
        TestDataPreprocessing,
        TestDataPipeline,
        TestIntelligentPipelineSelector,
        TestAutoModelTrainer,
        TestValidation,
        TestLogging,
        TestExceptions,
        TestConfiguration,
        TestEndToEnd,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your Intelligent AutoML Framework is ready for production!")
    else:
        print("\n⚠️  Some tests failed. Please review and fix issues.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)


        # Should have trained models
        self.assertGreater(len(trainer.trained_models), 0)
        self.assertIsNotNone(trainer.best_model)
        
        # Should be able to make predictions
        predictions = trainer.predict(X)
        self.assertEqual(len(predictions), len(y))
    
    def test_regression_training(self):
        """Test regression model training."""
        trainer = AutoModelTrainer(
            task_type='regression',
            models_to_try=['random_forest', 'linear_regression'],
            verbose=False
        )
        
        X = self.regression_data.drop('target', axis=1)
        y = self.regression_data['target']
        
        trainer.fit(X, y)

        self.assertGreater(len(trainer.trained_models), 0)
        self.assertIsNotNone(trainer.best_model)
        self.assertGreater(trainer.best_score, 0)
        
        # Should be able to make predictions
        predictions = trainer.predict(X)
        self.assertEqual(len(predictions), len(y))
    
    def test_regression_training(self):
        """Test regression model training."""
        trainer = AutoModelTrainer(
            task_type='regression',
            models_to_try=['random_forest', 'linear_regression'],
            verbose=False
        )
        
        X = self.regression_data.drop('target', axis=1)
        y = self.regression_data['target']
        
        trainer.fit(X, y)
        
        # Should have trained models