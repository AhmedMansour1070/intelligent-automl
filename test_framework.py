# test_framework.py

"""
Quick test to verify the framework imports and basic functionality work.
"""

def test_imports():
    """Test all important imports."""
    print("🧪 Testing Framework Imports...")
    
    try:
        # Test evaluation module
        from intelligent_automl.evaluation.multi_metric_evaluator import MultiMetricEvaluator
        print("✅ MultiMetricEvaluator imported successfully!")
        
        # Test models module  
        from intelligent_automl.models.auto_trainer import EnhancedAutoModelTrainer
        print("✅ EnhancedAutoModelTrainer imported successfully!")
        
        # Test main framework
        from intelligent_automl import IntelligentAutoMLFramework
        print("✅ IntelligentAutoMLFramework imported successfully!")
        
        # Test config
        from intelligent_automl.core.config import AutoMLConfig, DataConfig
        print("✅ Config classes imported successfully!")
        
        print("\n🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic framework functionality."""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_classification
        from intelligent_automl import IntelligentAutoMLFramework
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        df.to_csv('test_data.csv', index=False)
        
        print("📊 Created test dataset")
        
        # Initialize framework
        framework = IntelligentAutoMLFramework(verbose=False)
        print("✅ Framework initialized")
        
        # Test data loading
        framework.load_data('test_data.csv', 'target')
        print("✅ Data loaded")
        
        # Test preprocessing
        framework.preprocess_data()
        print("✅ Data preprocessed")
        
        # Test model training with limited models for speed
        framework.train_models(models_to_try=['random_forest'], n_trials=5)
        print("✅ Model trained")
        
        # Test prediction
        test_X = df.drop('target', axis=1).head(5)
        predictions = framework.make_predictions(test_X)
        print(f"✅ Predictions made: {len(predictions)} samples")
        
        # Clean up
        import os
        os.remove('test_data.csv')
        print("🗑️ Cleaned up test files")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_metric_evaluation():
    """Test multi-metric evaluation specifically."""
    print("\n🧪 Testing Multi-Metric Evaluation...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_classification
        from intelligent_automl.evaluation.multi_metric_evaluator import MultiMetricEvaluator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Create test data
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a simple model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Test multi-metric evaluation
        evaluator = MultiMetricEvaluator()
        metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
        
        print(f"✅ Evaluated {len(metrics.all_metrics)} metrics")
        print(f"🎯 Primary metric: {metrics.primary_metric} = {metrics.primary_score:.4f}")
        
        # Show top metrics
        print("📊 Top 3 metrics:")
        for i, metric in enumerate(metrics.get_best_metrics(3), 1):
            direction = "↗" if metric.higher_is_better else "↘"
            print(f"  {i}. {metric.name}: {metric.value:.4f} {direction}")
        
        print("\n🎉 Multi-metric evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-metric test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 INTELLIGENT AUTOML FRAMEWORK TESTING")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_multi_metric_evaluation():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 TESTING COMPLETE: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Framework is working correctly.")
        print("✅ You can now use the framework with multi-metric evaluation!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()