# ===================================
# FILE: automl_framework/data/preprocessors.py
# LOCATION: /automl_framework/automl_framework/data/preprocessors.py
# ===================================

"""
Data preprocessing components for the AutoML framework.

This module implements concrete data processors that handle missing values,
feature scaling, categorical encoding, and other data transformations.
"""
from ..core.base import DataProcessor
from ..core.exceptions import PreprocessingError, handle_sklearn_error

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer,
    OneHotEncoder, LabelEncoder, OrdinalEncoder, 
    PowerTransformer, KBinsDiscretizer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.compose import ColumnTransformer
import warnings
from ..core.base import DataProcessor
from ..core.exceptions import PreprocessingError, handle_sklearn_error
from ..core.types import DataFrame, Series, ScalingMethod, EncodingMethod


class MissingValueHandler(DataProcessor):
    """
    Handles missing values in datasets using various imputation strategies.
    
    Supports different strategies for numeric and categorical features,
    with automatic detection of column types.
    """
    
    def __init__(self, 
                 numeric_strategy: str = 'mean',
                 categorical_strategy: str = 'most_frequent',
                 fill_value: Any = None,
                 n_neighbors: int = 5):
        """
        Initialize the missing value handler.
        
        Args:
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'constant', 'knn')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
            fill_value: Fill value when using 'constant' strategy
            n_neighbors: Number of neighbors for KNN imputation
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        
        self.numeric_imputer: Optional[Union[SimpleImputer, KNNImputer]] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self._fitted = False
    
    @handle_sklearn_error
    def fit(self, data: DataFrame) -> 'MissingValueHandler':
        """
        Fit the imputers to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        # Identify column types
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit numeric imputer
        if self.numeric_columns:
            if self.numeric_strategy == 'knn':
                self.numeric_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            else:
                self.numeric_imputer = SimpleImputer(
                    strategy=self.numeric_strategy,
                    fill_value=self.fill_value
                )
            self.numeric_imputer.fit(data[self.numeric_columns])
        
        # Fit categorical imputer
        if self.categorical_columns:
            self.categorical_imputer = SimpleImputer(
                strategy=self.categorical_strategy,
                fill_value=self.fill_value
            )
            categorical_data = data[self.categorical_columns].copy()
            for col in self.categorical_columns:
                categorical_data[col] = categorical_data[col].astype(str).replace('nan', np.nan)
            self.categorical_imputer.fit(categorical_data)


        self._fitted = True
        return self
    
    @handle_sklearn_error
    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transform data by imputing missing values.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed
        """
        if not self._fitted:
            raise PreprocessingError("MissingValueHandler must be fitted before transform")
        
        data_copy = data.copy()
        
        # Transform numeric columns
        if self.numeric_columns and self.numeric_imputer is not None:
            imputed_numeric = self.numeric_imputer.transform(data_copy[self.numeric_columns])
            data_copy[self.numeric_columns] = imputed_numeric
        
        # Transform categorical columns
        if self.categorical_columns and self.categorical_imputer is not None:
            imputed_categorical = self.categorical_imputer.transform(data_copy[self.categorical_columns])
            data_copy[self.categorical_columns] = imputed_categorical
        
        return data_copy
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the processor."""
        return {
            'numeric_strategy': self.numeric_strategy,
            'categorical_strategy': self.categorical_strategy,
            'fill_value': self.fill_value,
            'n_neighbors': self.n_neighbors
        }
    
    def set_params(self, **params) -> 'MissingValueHandler':
        """Set parameters of the processor."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        self._fitted = False  # Need to refit after parameter change
        return self


class FeatureScaler(DataProcessor):
    """
    Scales numerical features using various scaling methods.
    
    Automatically identifies numeric columns and applies scaling
    while preserving non-numeric columns.
    """
    
    def __init__(self, method: str = 'minmax', **scaler_params):
        """
        Initialize the feature scaler.
        
        Args:
            method: Scaling method ('minmax', 'standard', 'robust', 'quantile')
            **scaler_params: Additional parameters for the scaler
        """
        self.method = method
        self.scaler_params = scaler_params
        self.scaler = None
        self.numeric_columns: List[str] = []
        self._fitted = False
        
        # Initialize scaler based on method
        self._init_scaler()
    
    def _init_scaler(self):
        """Initialize the appropriate scaler based on method."""
        scaler_map = {
            'minmax': MinMaxScaler,
            'standard': StandardScaler,
            'robust': RobustScaler,
            'quantile': QuantileTransformer
        }
        
        if self.method not in scaler_map:
            raise PreprocessingError(f"Unknown scaling method: {self.method}")
        
        self.scaler = scaler_map[self.method](**self.scaler_params)
    
    @handle_sklearn_error
    def fit(self, data: DataFrame) -> 'FeatureScaler':
        """
        Fit the scaler to numeric columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.numeric_columns:
            self.scaler.fit(data[self.numeric_columns])
        
        self._fitted = True
        return self
    
    @handle_sklearn_error
    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transform data by scaling numeric features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with scaled numeric features
        """
        if not self._fitted:
            raise PreprocessingError("FeatureScaler must be fitted before transform")
        
        if not self.numeric_columns:
            return data.copy()
        
        data_copy = data.copy()
        scaled_data = self.scaler.transform(data_copy[self.numeric_columns])
        data_copy[self.numeric_columns] = scaled_data
        
        return data_copy
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the processor."""
        params = {'method': self.method}
        params.update(self.scaler_params)
        return params
    
    def set_params(self, **params) -> 'FeatureScaler':
        """Set parameters of the processor."""
        if 'method' in params:
            self.method = params.pop('method')
            self._init_scaler()
        
        self.scaler_params.update(params)
        self._init_scaler()
        self._fitted = False
        return self


class CategoricalEncoder(DataProcessor):
    """
    Fixed categorical encoder that properly handles pandas categorical types.
    """
    
    def __init__(self, 
                 method: str = 'onehot',
                 handle_unknown: str = 'ignore',
                 drop_first: bool = False,
                 max_categories: Optional[int] = None):
        """
        Initialize the categorical encoder.
        
        Args:
            method: Encoding method ('onehot', 'label', 'ordinal')
            handle_unknown: How to handle unknown categories ('ignore', 'error')
            drop_first: Whether to drop first category in one-hot encoding
            max_categories: Maximum number of categories per feature
        """
        self.method = method
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first
        self.max_categories = max_categories
        
        self.encoder = None
        self.categorical_columns: List[str] = []
        self.encoded_feature_names: List[str] = []
        self._fitted = False
        
        # Initialize encoder
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize the appropriate encoder based on method."""
        if self.method == 'onehot':
            self.encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.handle_unknown,
                drop='first' if self.drop_first else None
            )
        elif self.method == 'label':
            # For label encoding, we'll use a dict of LabelEncoders
            self.encoder = {}
        elif self.method == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            raise PreprocessingError(f"Unknown encoding method: {self.method}")
    
    def _convert_categorical_to_object(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas categorical columns to object type for consistent handling."""
        data_copy = data.copy()
        
        for col in data_copy.columns:
            if pd.api.types.is_categorical_dtype(data_copy[col]):
                # Convert categorical to string, preserving NaN values
                data_copy[col] = data_copy[col].astype(str)
                # Replace 'nan' strings back to actual NaN
                data_copy[col] = data_copy[col].replace('nan', np.nan)
        
        return data_copy
    
    @handle_sklearn_error
    def fit(self, data: pd.DataFrame) -> 'CategoricalEncoder':
        """
        Fit the encoder to categorical columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        # Convert data for consistent handling
        data_processed = self._convert_categorical_to_object(data)
        
        # Find categorical columns
        self.categorical_columns = data_processed.select_dtypes(include=['object']).columns.tolist()
        
        if not self.categorical_columns:
            self._fitted = True
            return self
        
        # Filter categories based on max_categories
        if self.max_categories:
            filtered_columns = []
            for col in self.categorical_columns:
                if data_processed[col].nunique() <= self.max_categories:
                    filtered_columns.append(col)
            self.categorical_columns = filtered_columns
        
        if not self.categorical_columns:
            self._fitted = True
            return self
        
        # Fit encoder based on method
        if self.method == 'onehot':
            # Fill NaN values temporarily for fitting
            data_for_fitting = data_processed[self.categorical_columns].fillna('__MISSING__')
            self.encoder.fit(data_for_fitting)
            self.encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_columns).tolist()
            
        elif self.method == 'label':
            for col in self.categorical_columns:
                le = LabelEncoder()
                # Get unique values excluding NaN
                unique_values = data_processed[col].dropna().astype(str).unique()
                le.fit(unique_values)
                self.encoder[col] = le
                
        elif self.method == 'ordinal':
            # Fill NaN values temporarily for fitting
            data_for_fitting = data_processed[self.categorical_columns].fillna('__MISSING__')
            self.encoder.fit(data_for_fitting)
        
        self._fitted = True
        return self
    
    @handle_sklearn_error
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by encoding categorical features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        if not self._fitted:
            raise PreprocessingError("CategoricalEncoder must be fitted before transform")
        
        if not self.categorical_columns:
            return data.copy()
        
        # Convert data for consistent handling
        data_copy = self._convert_categorical_to_object(data)
        
        if self.method == 'onehot':
            # One-hot encoding
            data_for_encoding = data_copy[self.categorical_columns].fillna('__MISSING__')
            encoded_data = self.encoder.transform(data_for_encoding)
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=self.encoded_feature_names,
                index=data_copy.index
            )
            
            # Drop original categorical columns and add encoded ones
            data_copy = data_copy.drop(columns=self.categorical_columns)
            data_copy = pd.concat([data_copy, encoded_df], axis=1)
            
        elif self.method == 'label':
            # Label encoding
            for col in self.categorical_columns:
                # Handle unknown categories and NaN values
                col_data = data_copy[col].astype(str)
                
                # Create a mask for known categories
                known_mask = col_data.isin(self.encoder[col].classes_)
                nan_mask = data_copy[col].isna()
                
                # Initialize with -1 (unknown category marker)
                encoded_values = np.full(len(col_data), -1, dtype=int)
                
                # Encode known categories
                if known_mask.any():
                    encoded_values[known_mask] = self.encoder[col].transform(col_data[known_mask])
                
                # Handle NaN values (set to -2 to distinguish from unknown categories)
                if nan_mask.any():
                    encoded_values[nan_mask] = -2
                
                data_copy[col] = encoded_values
                
        elif self.method == 'ordinal':
            # Ordinal encoding
            data_for_encoding = data_copy[self.categorical_columns].fillna('__MISSING__')
            encoded_data = self.encoder.transform(data_for_encoding)
            data_copy[self.categorical_columns] = encoded_data
        
        return data_copy
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the processor."""
        return {
            'method': self.method,
            'handle_unknown': self.handle_unknown,
            'drop_first': self.drop_first,
            'max_categories': self.max_categories
        }
    
    def set_params(self, **params) -> 'CategoricalEncoder':
        """Set parameters of the processor."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        self._init_encoder()
        self._fitted = False
        return self

class OutlierHandler(DataProcessor):
    """
    Handles outliers in numeric features using various detection and treatment methods.
    
    Supports IQR, Z-score, and Isolation Forest methods for outlier detection,
    with options to remove, cap, or transform outliers.
    """
    
    def __init__(self, 
                 method: str = 'iqr',
                 treatment: str = 'cap',
                 threshold: float = 1.5,
                 contamination: float = 0.1):
        """
        Initialize the outlier handler.
        
        Args:
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            treatment: Treatment method ('remove', 'cap', 'transform')
            threshold: Threshold for IQR or Z-score methods
            contamination: Expected proportion of outliers for Isolation Forest
        """
        self.method = method
        self.treatment = treatment
        self.threshold = threshold
        self.contamination = contamination
        
        self.numeric_columns: List[str] = []
        self.lower_bounds: Dict[str, float] = {}
        self.upper_bounds: Dict[str, float] = {}
        self._fitted = False
    
    @handle_sklearn_error
    def fit(self, data: DataFrame) -> 'OutlierHandler':
        """
        Fit the outlier handler to compute bounds.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.numeric_columns:
            self._fitted = True
            return self
        
        for col in self.numeric_columns:
            if self.method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds[col] = Q1 - self.threshold * IQR
                self.upper_bounds[col] = Q3 + self.threshold * IQR
                
            elif self.method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                self.lower_bounds[col] = mean - self.threshold * std
                self.upper_bounds[col] = mean + self.threshold * std
        
        self._fitted = True
        return self
    
    @handle_sklearn_error
    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transform data by handling outliers.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        if not self._fitted:
            raise PreprocessingError("OutlierHandler must be fitted before transform")
        
        if not self.numeric_columns:
            return data.copy()
        
        data_copy = data.copy()
        
        for col in self.numeric_columns:
            if col in self.lower_bounds:
                if self.treatment == 'cap':
                    data_copy[col] = data_copy[col].clip(
                        lower=self.lower_bounds[col],
                        upper=self.upper_bounds[col]
                    )
                elif self.treatment == 'remove':
                    # Mark outliers for removal (will be handled by pipeline)
                    mask = (
                        (data_copy[col] >= self.lower_bounds[col]) &
                        (data_copy[col] <= self.upper_bounds[col])
                    )
                    data_copy = data_copy[mask]
        
        return data_copy
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the processor."""
        return {
            'method': self.method,
            'treatment': self.treatment,
            'threshold': self.threshold,
            'contamination': self.contamination
        }
    
    def set_params(self, **params) -> 'OutlierHandler':
        """Set parameters of the processor."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        self._fitted = False
        return self


class FeatureEngineering(DataProcessor):
    """
    Creates new features through polynomial features, interactions, and transformations.
    
    Supports polynomial feature creation, interaction terms,
    and mathematical transformations of existing features.
    """
    
    def __init__(self, 
                 polynomial_degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = False,
                 log_transform: bool = False,
                 sqrt_transform: bool = False):
        """
        Initialize the feature engineering processor.
        
        Args:
            polynomial_degree: Degree for polynomial features
            interaction_only: Only create interaction terms, no powers
            include_bias: Include bias column in polynomial features
            log_transform: Apply log transformation to positive features
            sqrt_transform: Apply square root transformation
        """
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.log_transform = log_transform
        self.sqrt_transform = sqrt_transform
        
        self.numeric_columns: List[str] = []
        self.log_columns: List[str] = []
        self.sqrt_columns: List[str] = []
        self._fitted = False
    
    @handle_sklearn_error
    def fit(self, data: DataFrame) -> 'FeatureEngineering':
        """
        Fit the feature engineering processor.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.log_transform:
            # Identify columns suitable for log transformation (positive values)
            self.log_columns = [
                col for col in self.numeric_columns 
                if (data[col] > 0).all()
            ]
        
        if self.sqrt_transform:
            # Identify columns suitable for sqrt transformation (non-negative values)
            self.sqrt_columns = [
                col for col in self.numeric_columns 
                if (data[col] >= 0).all()
            ]
        
        self._fitted = True
        return self
    
    @handle_sklearn_error
    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transform data by engineering new features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if not self._fitted:
            raise PreprocessingError("FeatureEngineering must be fitted before transform")
        
        data_copy = data.copy()
        
        # Log transformation
        if self.log_transform and self.log_columns:
            for col in self.log_columns:
                data_copy[f'{col}_log'] = np.log1p(data_copy[col])
        
        # Square root transformation
        if self.sqrt_transform and self.sqrt_columns:
            for col in self.sqrt_columns:
                data_copy[f'{col}_sqrt'] = np.sqrt(data_copy[col])
        
        return data_copy
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the processor."""
        return {
            'polynomial_degree': self.polynomial_degree,
            'interaction_only': self.interaction_only,
            'include_bias': self.include_bias,
            'log_transform': self.log_transform,
            'sqrt_transform': self.sqrt_transform
        }
    
    def set_params(self, **params) -> 'FeatureEngineering':
        """Set parameters of the processor."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        self._fitted = False
        return self



class DateTimeProcessor(DataProcessor):
    """
    Processes datetime columns by extracting useful features.
    
    Extracts year, month, day, hour, minute, weekday, quarter,
    and other temporal features from datetime columns.
    """
    
    def __init__(self, 
                 datetime_columns: Optional[List[str]] = None,
                 extract_components: bool = True,
                 extract_cyclical: bool = True,
                 drop_original: bool = True):
        """
        Initialize the datetime processor.
        
        Args:
            datetime_columns: List of datetime column names (auto-detect if None)
            extract_components: Extract basic components (year, month, etc.)
            extract_cyclical: Extract cyclical features (sin/cos of time components)
            drop_original: Drop original datetime columns after processing
        """
        self.datetime_columns = datetime_columns
        self.extract_components = extract_components
        self.extract_cyclical = extract_cyclical
        self.drop_original = drop_original
        
        self.detected_datetime_columns: List[str] = []
        self._fitted = False
    
    def _detect_datetime_columns(self, data: DataFrame) -> List[str]:
        """Detect datetime columns in the DataFrame with improved error handling."""
        datetime_cols = []
        
        # Check existing datetime columns
        datetime_cols.extend(data.select_dtypes(include=['datetime64']).columns.tolist())
        
        # Try to parse object columns as datetime
        for col in data.select_dtypes(include=['object']).columns:
            try:
                sample = data[col].dropna().head(100)
                if len(sample) == 0:
                    continue
                
                # Suppress warnings during datetime detection
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # First try: strict parsing
                    try:
                        pd.to_datetime(sample, errors='raise')
                        datetime_cols.append(col)
                        continue
                    except:
                        pass
                    
                    # Second try: infer format
                    try:
                        parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                        # If more than 80% of values were successfully parsed, consider it a datetime column
                        if (parsed.notna().sum() / len(sample)) > 0.8:
                            datetime_cols.append(col)
                            continue
                    except:
                        pass
                        
            except:
                # Check for common date patterns in column names as fallback
                date_indicators = ['date', 'time', 'created', 'updated', 'timestamp', '_at', '_on']
                if any(indicator in col.lower() for indicator in date_indicators):
                    try:
                        sample = data[col].dropna().head(10)
                        if len(sample) > 0:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                parsed = pd.to_datetime(sample, errors='coerce')
                                if not parsed.isna().all():
                                    datetime_cols.append(col)
                    except:
                        pass
        
        return list(set(datetime_cols))  # Remove duplicates
    
    @handle_sklearn_error
    def fit(self, data: DataFrame) -> 'DateTimeProcessor':
        """
        Fit the datetime processor.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        if self.datetime_columns is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.detected_datetime_columns = self._detect_datetime_columns(data)
        else:
            self.detected_datetime_columns = self.datetime_columns
        
        self._fitted = True
        return self
    
    @handle_sklearn_error
    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transform data by extracting datetime features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with datetime features extracted
        """
        if not self._fitted:
            raise PreprocessingError("DateTimeProcessor must be fitted before transform")
        
        if not self.detected_datetime_columns:
            return data.copy()
        
        data_copy = data.copy()
        
        # Suppress all warnings during datetime processing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for col in self.detected_datetime_columns:
                if col not in data_copy.columns:
                    continue
                    
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data_copy[col]):
                    try:
                        data_copy[col] = pd.to_datetime(data_copy[col], errors='coerce')
                    except:
                        # If conversion fails, skip this column
                        continue
                
                dt_col = data_copy[col]
                
                # Skip if all values are NaT after conversion
                if dt_col.isna().all():
                    continue
                
                if self.extract_components:
                    # Basic components
                    try:
                        data_copy[f'{col}_year'] = dt_col.dt.year
                        data_copy[f'{col}_month'] = dt_col.dt.month
                        data_copy[f'{col}_day'] = dt_col.dt.day
                        data_copy[f'{col}_hour'] = dt_col.dt.hour
                        data_copy[f'{col}_minute'] = dt_col.dt.minute
                        data_copy[f'{col}_weekday'] = dt_col.dt.weekday
                        data_copy[f'{col}_quarter'] = dt_col.dt.quarter
                        data_copy[f'{col}_is_weekend'] = (dt_col.dt.weekday >= 5).astype(int)
                    except Exception:
                        # If any component extraction fails, skip to next column
                        continue
                    
                if self.extract_cyclical:
                    # Cyclical features for better ML performance
                    try:
                        data_copy[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_col.dt.month / 12)
                        data_copy[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_col.dt.month / 12)
                        data_copy[f'{col}_day_sin'] = np.sin(2 * np.pi * dt_col.dt.day / 31)
                        data_copy[f'{col}_day_cos'] = np.cos(2 * np.pi * dt_col.dt.day / 31)
                        data_copy[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
                        data_copy[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
                    except Exception:
                        # If cyclical feature extraction fails, continue without them
                        pass
        
        # Drop original datetime columns if requested
        if self.drop_original:
            columns_to_drop = [col for col in self.detected_datetime_columns if col in data_copy.columns]
            data_copy = data_copy.drop(columns=columns_to_drop)
        
        return data_copy
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the processor."""
        return {
            'datetime_columns': self.datetime_columns,
            'extract_components': self.extract_components,
            'extract_cyclical': self.extract_cyclical,
            'drop_original': self.drop_original
        }
    
    def set_params(self, **params) -> 'DateTimeProcessor':
        """Set parameters of the processor."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        self._fitted = False
        return self    


class FeatureSelector(DataProcessor):
    """
    Selects the most important features using various selection methods.
    
    Supports univariate selection, recursive feature elimination,
    and other feature selection techniques.
    """
    
    def __init__(self, 
                 method: str = 'mutual_info',
                 k: int = 10,
                 threshold: Optional[float] = None):
        """
        Initialize the feature selector.
        
        Args:
            method: Selection method ('mutual_info', 'chi2', 'f_test')
            k: Number of features to select
            threshold: Threshold for feature scores (if not using k)
        """
        self.method = method
        self.k = k
        self.threshold = threshold
        
        self.selector = None
        self.selected_features: List[str] = []
        self._fitted = False
    
    @handle_sklearn_error
    def fit(self, data: DataFrame, target: Optional[Series] = None) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            data: Input DataFrame
            target: Target variable (required for supervised selection)
            
        Returns:
            Self for method chaining
        """
        if target is None:
            raise PreprocessingError("Target variable required for feature selection")
        
        # Select only numeric columns for feature selection
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            self._fitted = True
            return self
        
        # Choose selection method
        if self.method == 'mutual_info':
            # Determine if classification or regression
            if target.dtype == 'object' or target.nunique() < 20:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        else:
            raise PreprocessingError(f"Unknown selection method: {self.method}")
        
        # Initialize selector
        if self.threshold is not None:
            # Use threshold-based selection
            from sklearn.feature_selection import SelectPercentile
            self.selector = SelectPercentile(score_func=score_func, percentile=self.threshold)
        else:
            # Use k-best selection
            self.selector = SelectKBest(score_func=score_func, k=min(self.k, numeric_data.shape[1]))
        
        # Fit selector
        self.selector.fit(numeric_data, target)
        
        # Get selected feature names
        selected_mask = self.selector.get_support()
        self.selected_features = numeric_data.columns[selected_mask].tolist()
        
        # Add non-numeric columns to selected features
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        self.selected_features.extend(non_numeric_columns)
        
        self._fitted = True
        return self
    
    @handle_sklearn_error
    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transform data by selecting features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        if not self._fitted:
            raise PreprocessingError("FeatureSelector must be fitted before transform")
        
        # Return only selected features
        available_features = [col for col in self.selected_features if col in data.columns]
        return data[available_features].copy()
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the processor."""
        return {
            'method': self.method,
            'k': self.k,
            'threshold': self.threshold
        }
    
    def set_params(self, **params) -> 'FeatureSelector':
        """Set parameters of the processor."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        self._fitted = False
        return self
    
    def get_feature_scores(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to scores
        """
        if not self._fitted or self.selector is None:
            return {}
        
        numeric_data_columns = self.selector.feature_names_in_ if hasattr(self.selector, 'feature_names_in_') else []
        scores = self.selector.scores_ if hasattr(self.selector, 'scores_') else []
        
        if len(numeric_data_columns) == len(scores):
            return dict(zip(numeric_data_columns, scores))
        
        return {}