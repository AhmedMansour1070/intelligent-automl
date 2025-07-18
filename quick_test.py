#!/usr/bin/env python3
"""
quick_test.py - Quick test script for Intelligent AutoML Framework

This script provides a comprehensive test of the framework to ensure
everything is working correctly after installation.
"""

import sys
import time
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test main imports
        import intelligent_automl
        print(f"  ✅ Main package: v{intelligent_automl.__version__}")
        
        from intelligent_automl import IntelligentAutoMLFramework
        print("  ✅ Main framework class")
        
        from intelligent_automl import create_intelligent_pipeline
        print("  ✅ Pipeline creation function")
        
        from intelligent_automl.intelligence import IntelligentPipelineSelector
        print("  ✅ Intelligence module")
        
        from intelligent_automl.data import DataPipeline
        print("  ✅ Data processing components")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from intelligent_automl import create_intelligent_pipeline
        
        # Create sample data
        np.random.seed(42)
        data = {
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.exponential(1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        
        # Add some missing values
        df.loc[::10, 'numeric1'] = np.nan
        df.loc[::15, 'category'] = np.nan
        
        print(f"  📊 Created test data: {df.shape}")
        
        # Test pipeline creation
        pipeline = create_intelligent_pipeline(df, target_column='target')
        print(f"  ✅ Pipeline created with {len(pipeline)} steps")
        
        # Test preprocessing
        features = df.drop('target', axis=1)
        processed = pipeline.fit_transform(features)
        print(f"  ✅ Preprocessing: {features.shape} → {processed.shape}")
        
        # Verify data quality
        missing_after = processed.isnull().sum().sum()
        print(f"  ✅ Missing values: {missing_after} (should be 0)")
        
        return True
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_complete_framework():
    """Test the complete AutoML framework."""
    print("\n🧪 Testing complete AutoML framework...")
    
    try:
        import pandas as pd
        import numpy as np
        from intelligent_automl import IntelligentAutoMLFramework
        
        # Create sample data
        np.random.seed(42)
        data = {
            'age': np.random.normal(35, 10, 200),
            'income': np.random.exponential(50000, 200),
            'city': np.random.choice(['NYC', 'LA', 'Chicago'], 200),
            'target': np.