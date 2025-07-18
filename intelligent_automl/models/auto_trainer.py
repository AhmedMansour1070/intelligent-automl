#!/usr/bin/env python
"""
Intelligent model training module for AutoML framework.

This module provides automatic model selection, training, and optimization
with intelligent defaults and performance monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import warnings
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..core.base import ModelStrategy, Evaluator, HyperparameterOptimizer
from ..core.exceptions import ModelTrainingError, OptimizationError
from ..core.types import TaskType, ModelType, DataFrame, Series, Metrics


@dataclass
class ModelPerformance:
    """Performance metrics for a trained model."""
    model_name: str
    task_type: str
    train_score: float
    val_score: float
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    memory_usage: float
    feature_importance: Optional[Dict[str, float]] = None


class AutoModelTrainer:
    """
    Intelligent model trainer that automatically selects and trains
    the best model for a given dataset and task.
    """
    
    def __init__(self, 
                 task_type: Optional[str] = None,
                 time_limit_minutes: Optional[int] = None,
                 models_to_try: Optional[List[str]] = None,
                 optimization_enabled: bool = True,
                 cross_validation_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the auto model trainer.
        
        Args:
            task_type: 'classification' or 'regression' (auto-detect if None)
            time_limit_minutes: Maximum training time in minutes
            models_to_try: List of model names to try
            optimization_enabled: Whether to perform hyperparameter optimization
            cross_validation_folds: Number of CV folds
            test_size: Test split size
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.task_type = task_type
        self.time_limit_minutes = time_limit_minutes
        self.models_to_try = models_to_try or self._get_default_models()
        self.optimization_enabled = optimization_enabled
        self.cv_folds = cross_validation_folds
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        self.trained_models: Dict[str, Any] = {}
        self.model_performances: List[ModelPerformance] = []
        self.best_model = None
        self.best_score = -np.inf
        self.training_start_time = None
        
    def _get_default_models(self) -> List[str]:
        """Get default models to try based on task type."""
        return [
            'random_forest',
            'logistic_regression',
            'linear_regression',
            'svm'
        ]
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """
        Improved automatic task type detection.
        
        Args:
            y: Target variable
            
        Returns:
            'classification' or 'regression'
        """
        if self.task_type:
            return self.task_type
        
        # First check: is it numeric?
        if not pd.api.types.is_numeric_dtype(y):
            return 'classification'
        
        # Get basic statistics
        unique_count = y.nunique()
        total_count = len(y)
        unique_ratio = unique_count / total_count
        
        # Check column name for regression indicators
        column_name = y.name.lower() if y.name else ''
        regression_keywords = [
            'duration', 'time', 'price', 'amount', 'value', 'score', 
            'rate', 'distance', 'length', 'height', 'weight', 'size',
            'cost', 'revenue', 'sales', 'income', 'age', 'speed'
        ]
        
        # Strong regression indicators
        if any(keyword in column_name for keyword in regression_keywords):
            return 'regression'
        
        # Rule 1: Binary classification (exactly 2 unique values)
        if unique_count == 2:
            return 'classification'
        
        # Rule 2: Very few unique values relative to data size
        if unique_count <= 20 and unique_ratio < 0.01:  # Less than 1% unique
            return 'classification'
        
        # Rule 3: High unique ratio suggests regression
        if unique_ratio > 0.05:  # More than 5% unique values
            return 'regression'
        
        # Rule 4: Large number of unique values
        if unique_count > 50:
            return 'regression'
        
        # Rule 5: Check data range and distribution
        if y.dtype in ['float64', 'float32']:
            # Float types are more likely regression
            return 'regression'
        
        # Rule 6: For integers, check if they look continuous
        if y.dtype in ['int64', 'int32']:
            y_clean = y.dropna()
            if len(y_clean) > 0:
                data_range = y_clean.max() - y_clean.min()
                if data_range > 100 and unique_count > 20:
                    return 'regression'
        
        # Default: if unsure and many unique values, assume regression
        if unique_count > 10:
            return 'regression'
        
        # Final fallback
        return 'classification' 
    def _get_model_instance(self, model_name: str, task_type: str) -> Any:
        """Get a model instance based on name and task type."""
        models = {
            'random_forest': {
                'classification': RandomForestClassifier(random_state=self.random_state),
                'regression': RandomForestRegressor(random_state=self.random_state)
            },
            'logistic_regression': {
                'classification': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'regression': LinearRegression()
            },
            'linear_regression': {
                'classification': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'regression': LinearRegression()
            },
            'svm': {
                'classification': SVC(random_state=self.random_state, probability=True),
                'regression': SVR()
            }
        }
        
        if model_name not in models:
            raise ModelTrainingError(f"Unknown model: {model_name}")
        
        return models[model_name][task_type]
    
    def _get_scoring_metric(self, task_type: str) -> str:
        """Get appropriate scoring metric for task type."""
        if task_type == 'classification':
            return 'accuracy'
        else:
            return 'r2'
    
    def _evaluate_model(self, model: Any, X: DataFrame, y: Series, 
                       task_type: str) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        predictions = model.predict(X)
        
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y, predictions, average='weighted', zero_division=0),
                'f1': f1_score(y, predictions, average='weighted', zero_division=0)
            }
        else:
            metrics = {
                'r2': r2_score(y, predictions),
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions))
            }
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficient values
                coef = np.abs(model.coef_).flatten() if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                return dict(zip(feature_names, coef))
        except Exception:
            pass
        
        return None
    
    def _train_single_model(self, model_name: str, X_train: DataFrame, y_train: Series,
                           X_val: DataFrame, y_val: Series, task_type: str) -> ModelPerformance:
        """Train a single model and return performance metrics."""
        if self.verbose:
            print(f"  🔧 Training {model_name}...")
        
        start_time = time.time()
        
        try:
            # Get model instance
            model = self._get_model_instance(model_name, task_type)
            
            # Train model
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Evaluate on validation set
            start_pred_time = time.time()
            val_metrics = self._evaluate_model(model, X_val, y_val, task_type)
            prediction_time = time.time() - start_pred_time
            
            # Cross-validation score
            scoring = self._get_scoring_metric(task_type)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=self.cv_folds, 
                    scoring=scoring,
                    n_jobs=-1
                )
            
            # Get primary score
            primary_metric = scoring
            val_score = val_metrics.get(primary_metric, 0)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model, X_train.columns.tolist())
            
            # Store model
            self.trained_models[model_name] = model
            
            # Memory usage (rough estimate)
            memory_usage = sum(np.array(attr).nbytes for attr in model.__dict__.values() 
                             if hasattr(attr, 'nbytes')) / 1024**2  # MB
            
            performance = ModelPerformance(
                model_name=model_name,
                task_type=task_type,
                train_score=self._evaluate_model(model, X_train, y_train, task_type)[primary_metric],
                val_score=val_score,
                cv_mean=cv_scores.mean(),
                cv_std=cv_scores.std(),
                training_time=training_time,
                prediction_time=prediction_time,
                memory_usage=memory_usage,
                feature_importance=feature_importance
            )
            
            if self.verbose:
                print(f"    ✅ {model_name}: {primary_metric}={val_score:.4f} (CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f})")
            
            return performance
            
        except Exception as e:
            if self.verbose:
                print(f"    ❌ {model_name} failed: {str(e)}")
            raise ModelTrainingError(f"Failed to train {model_name}: {str(e)}")
    
    def fit(self, X: DataFrame, y: Series) -> 'AutoModelTrainer':
        """
        Train multiple models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        self.training_start_time = time.time()
        
        if self.verbose:
            print(f"🚀 Starting intelligent model training...")
            print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Detect task type
        task_type = self._detect_task_type(y)
        if self.verbose:
            print(f"🎯 Task type: {task_type}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y if task_type == 'classification' and y.nunique() > 1 else None
        )
        
        if self.verbose:
            print(f"📈 Training split: {len(X_train)} train, {len(X_val)} validation")
            print(f"🔧 Models to try: {len(self.models_to_try)}")
        
        # Train each model
        self.model_performances = []
        
        for model_name in self.models_to_try:
            try:
                # Check time limit
                if self.time_limit_minutes:
                    elapsed = (time.time() - self.training_start_time) / 60
                    if elapsed > self.time_limit_minutes:
                        if self.verbose:
                            print(f"⏰ Time limit reached ({elapsed:.1f} minutes)")
                        break
                
                performance = self._train_single_model(
                    model_name, X_train, y_train, X_val, y_val, task_type
                )
                self.model_performances.append(performance)
                
                # Update best model
                if performance.val_score > self.best_score:
                    self.best_score = performance.val_score
                    self.best_model = self.trained_models[model_name]
                    
            except ModelTrainingError as e:
                if self.verbose:
                    print(f"⚠️  Skipping {model_name}: {str(e)}")
                continue
        
        total_time = time.time() - self.training_start_time
        
        if self.verbose:
            print(f"\n✅ Training complete in {total_time:.1f} seconds")
            print(f"🏆 Best model: {self.get_best_model_name()} (score: {self.best_score:.4f})")
        
        return self
    
    def predict(self, X: DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ModelTrainingError("No model has been trained yet")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities if available."""
        if self.best_model is None:
            raise ModelTrainingError("No model has been trained yet")
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        
        return None
    
    def get_best_model_name(self) -> str:
        """Get the name of the best performing model."""
        if not self.model_performances:
            return "None"
        
        best_performance = max(self.model_performances, key=lambda x: x.val_score)
        return best_performance.model_name
    
    def get_model_rankings(self) -> List[Dict[str, Any]]:
        """Get all models ranked by performance."""
        if not self.model_performances:
            return []
        
        rankings = []
        for perf in sorted(self.model_performances, key=lambda x: x.val_score, reverse=True):
            rankings.append({
                'model': perf.model_name,
                'score': perf.val_score,
                'cv_mean': perf.cv_mean,
                'cv_std': perf.cv_std,
                'training_time': perf.training_time,
                'memory_usage': perf.memory_usage
            })
        
        return rankings
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the best model."""
        if not self.model_performances:
            return None
        
        best_performance = max(self.model_performances, key=lambda x: x.val_score)
        return best_performance.feature_importance
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.model_performances:
            return {}
        
        best_model = self.get_best_model_name()
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        return {
            'best_model': best_model,
            'best_score': self.best_score,
            'models_trained': len(self.model_performances),
            'total_training_time': total_time,
            'model_rankings': self.get_model_rankings(),
            'feature_importance': self.get_feature_importance()
        }


def train_best_model(X: DataFrame, y: Series, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function to train the best model for a dataset.
    
    Args:
        X: Feature matrix
        y: Target variable
        **kwargs: Additional arguments for AutoModelTrainer
        
    Returns:
        Tuple of (best_model, training_summary)
    """
    trainer = AutoModelTrainer(**kwargs)
    trainer.fit(X, y)
    
    return trainer.best_model, trainer.get_training_summary()


# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_classification, make_regression
    
    print("🧪 Testing AutoModelTrainer...")
    
    # Classification example
    X_class, y_class = make_classification(n_samples=1000, n_features=20, n_redundant=5, random_state=42)
    X_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(X_class.shape[1])])
    y_class = pd.Series(y_class)
    
    print("\n📊 Classification Task:")
    trainer_class = AutoModelTrainer(task_type='classification')
    trainer_class.fit(X_class, y_class)
    
    summary = trainer_class.get_training_summary()
    print(f"Best model: {summary['best_model']}")
    print(f"Best score: {summary['best_score']:.4f}")
    
    # Regression example
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
    y_reg = pd.Series(y_reg)
    
    print("\n📈 Regression Task:")
    trainer_reg = AutoModelTrainer(task_type='regression')
    trainer_reg.fit(X_reg, y_reg)
    
    summary = trainer_reg.get_training_summary()
    print(f"Best model: {summary['best_model']}")
    print(f"Best score: {summary['best_score']:.4f}")
    
    print("\n✅ AutoModelTrainer testing complete!")