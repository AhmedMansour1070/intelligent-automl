# ===================================
# FILE: automl_framework/core/config.py
# LOCATION: /automl_framework/automl_framework/core/config.py
# ===================================

"""
Configuration management for the AutoML framework.

This module provides type-safe configuration classes using dataclasses
and validation logic to ensure configurations are valid.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class DataConfig:
    """Configuration for data handling and loading."""
    
    file_path: str
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    validation_size: float = 0.1
    stratify: bool = True
    handle_missing_target: str = 'drop'  # 'drop' or 'impute'
    date_columns: Optional[List[str]] = None
    index_column: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.test_size < 1:
            raise ConfigurationError("test_size must be between 0 and 1")
        
        if not 0 <= self.validation_size < 1:
            raise ConfigurationError("validation_size must be between 0 and 1")
        
        if self.test_size + self.validation_size >= 1:
            raise ConfigurationError("test_size + validation_size must be < 1")
        
        if not Path(self.file_path).exists():
            raise ConfigurationError(f"Data file not found: {self.file_path}")
        
        if self.handle_missing_target not in ['drop', 'impute']:
            raise ConfigurationError("handle_missing_target must be 'drop' or 'impute'")


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    scaling_method: str = 'minmax'  # 'minmax', 'standard', 'robust', 'none'
    encoding_method: str = 'onehot'  # 'onehot', 'label', 'target', 'none'
    handle_missing: str = 'auto'  # 'auto', 'mean', 'median', 'mode', 'drop'
    handle_outliers: bool = False
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    feature_selection: bool = False
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'chi2', 'f_test'
    feature_selection_k: int = 10
    polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_features: bool = False
    
    def __post_init__(self):
        """Validate preprocessing configuration."""
        valid_scaling = ['minmax', 'standard', 'robust', 'none']
        if self.scaling_method not in valid_scaling:
            raise ConfigurationError(f"scaling_method must be one of {valid_scaling}")
        
        valid_encoding = ['onehot', 'label', 'target', 'none']
        if self.encoding_method not in valid_encoding:
            raise ConfigurationError(f"encoding_method must be one of {valid_encoding}")
        
        valid_missing = ['auto', 'mean', 'median', 'mode', 'drop']
        if self.handle_missing not in valid_missing:
            raise ConfigurationError(f"handle_missing must be one of {valid_missing}")
        
        valid_outlier = ['iqr', 'zscore', 'isolation_forest']
        if self.outlier_method not in valid_outlier:
            raise ConfigurationError(f"outlier_method must be one of {valid_outlier}")
        
        if self.polynomial_degree < 1 or self.polynomial_degree > 5:
            raise ConfigurationError("polynomial_degree must be between 1 and 5")


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    cross_validation_folds: int = 5
    cross_validation_strategy: str = 'kfold'  # 'kfold', 'stratified', 'time_series'
    ensemble_method: Optional[str] = None  # 'voting', 'stacking', 'blending'
    ensemble_models: Optional[List[str]] = None
    auto_ensemble: bool = False
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.cross_validation_folds < 2:
            raise ConfigurationError("cross_validation_folds must be >= 2")
        
        valid_cv_strategies = ['kfold', 'stratified', 'time_series']
        if self.cross_validation_strategy not in valid_cv_strategies:
            raise ConfigurationError(f"cv_strategy must be one of {valid_cv_strategies}")
        
        if self.ensemble_method:
            valid_ensemble = ['voting', 'stacking', 'blending']
            if self.ensemble_method not in valid_ensemble:
                raise ConfigurationError(f"ensemble_method must be one of {valid_ensemble}")


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    enabled: bool = False
    method: str = 'random'  # 'grid', 'random', 'bayesian', 'optuna'
    n_trials: int = 50
    timeout_minutes: Optional[int] = None
    metric: str = 'accuracy'  # for classification: 'accuracy', 'f1', 'roc_auc'
                              # for regression: 'r2', 'mse', 'mae'
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    search_space: Dict[str, Any] = field(default_factory=dict)
    early_stopping: bool = True
    early_stopping_rounds: int = 10
    
    def __post_init__(self):
        """Validate optimization configuration."""
        valid_methods = ['grid', 'random', 'bayesian', 'optuna']
        if self.method not in valid_methods:
            raise ConfigurationError(f"optimization method must be one of {valid_methods}")
        
        if self.n_trials < 1:
            raise ConfigurationError("n_trials must be >= 1")
        
        valid_directions = ['maximize', 'minimize']
        if self.direction not in valid_directions:
            raise ConfigurationError(f"direction must be one of {valid_directions}")


@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    
    max_time_minutes: Optional[int] = None
    early_stopping: bool = True
    early_stopping_patience: int = 10
    save_intermediate: bool = False
    checkpoint_frequency: int = 10  # epochs
    verbose: bool = True
    log_level: str = 'INFO'
    random_seed: int = 42
    n_jobs: int = -1  # number of parallel jobs
    memory_limit_gb: Optional[float] = None
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.max_time_minutes is not None and self.max_time_minutes <= 0:
            raise ConfigurationError("max_time_minutes must be positive")
        
        if self.early_stopping_patience < 1:
            raise ConfigurationError("early_stopping_patience must be >= 1")
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of {valid_log_levels}")
        
        if self.checkpoint_frequency < 1:
            raise ConfigurationError("checkpoint_frequency must be >= 1")


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    metrics: List[str] = field(default_factory=lambda: ['accuracy'])
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    cross_validation: bool = True
    holdout_evaluation: bool = True
    bootstrap_samples: int = 0  # 0 means no bootstrap
    confidence_level: float = 0.95
    save_predictions: bool = True
    save_feature_importance: bool = True
    save_learning_curves: bool = False
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        if not self.metrics:
            raise ConfigurationError("At least one metric must be specified")
        
        if not 0 < self.confidence_level < 1:
            raise ConfigurationError("confidence_level must be between 0 and 1")
        
        if self.bootstrap_samples < 0:
            raise ConfigurationError("bootstrap_samples must be >= 0")


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_path: str = 'automl.log'
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    rotate_logs: bool = True
    max_log_size_mb: int = 10
    backup_count: int = 5
    track_metrics: bool = True
    metrics_backend: str = 'local'  # 'local', 'mlflow', 'wandb'
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.log_level not in valid_levels:
            raise ConfigurationError(f"log_level must be one of {valid_levels}")
        
        valid_backends = ['local', 'mlflow', 'wandb']
        if self.metrics_backend not in valid_backends:
            raise ConfigurationError(f"metrics_backend must be one of {valid_backends}")


@dataclass
class AutoMLConfig:
    """Main configuration class that combines all sub-configurations."""
    
    data: DataConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_type='random_forest'))
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoMLConfig':
        """Create configuration from dictionary."""
        return cls(
            data=DataConfig(**config_dict['data']),
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            model=ModelConfig(**config_dict['model']),
            optimization=OptimizationConfig(**config_dict.get('optimization', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'AutoMLConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'AutoMLConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """
        Validate the entire configuration and return any issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Check if model type is supported for ensemble
            if self.model.ensemble_method and not self.model.ensemble_models:
                errors.append("ensemble_models must be specified when using ensemble_method")
            
            # Check optimization metric compatibility
            if self.optimization.enabled:
                classification_metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
                regression_metrics = ['r2', 'mse', 'mae', 'rmse']
                
                # This would need task type detection logic
                # For now, just check if metric is known
                all_metrics = classification_metrics + regression_metrics
                if self.optimization.metric not in all_metrics:
                    errors.append(f"Unknown optimization metric: {self.optimization.metric}")
            
            # Check evaluation metrics
            known_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
                           'r2', 'mse', 'mae', 'rmse']
            for metric in self.evaluation.metrics:
                if metric not in known_metrics:
                    errors.append(f"Unknown evaluation metric: {metric}")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters for the configured model type."""
        base_params = self.model.hyperparameters.copy()
        base_params.update({
            'random_state': self.training.random_seed,
            'n_jobs': self.training.n_jobs if self.training.n_jobs != -1 else None
        })
        return base_params
    
    def is_classification_task(self) -> Optional[bool]:
        """
        Determine if this is a classification task based on metrics.
        
        Returns:
            True if classification, False if regression, None if ambiguous
        """
        classification_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
        regression_metrics = {'r2', 'mse', 'mae', 'rmse'}
        
        eval_metrics = set(self.evaluation.metrics)
        opt_metric = {self.optimization.metric} if self.optimization.enabled else set()
        all_metrics = eval_metrics | opt_metric
        
        has_classification = bool(all_metrics & classification_metrics)
        has_regression = bool(all_metrics & regression_metrics)
        
        if has_classification and not has_regression:
            return True
        elif has_regression and not has_classification:
            return False
        else:
            return None  # Ambiguous or no clear indication