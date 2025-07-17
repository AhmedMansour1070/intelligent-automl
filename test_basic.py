
def test_basic_imports():
    """Test that we can import the basic components."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test core imports
        from intelligent_automl.core import DataProcessor, ModelStrategy
        print("✅ Core base classes imported")
        
        from intelligent_automl.core import AutoMLError
        print("✅ Core exceptions imported")
        
        # Test data imports  
        from intelligent_automl.data import MissingValueHandler, DataPipeline
        print("✅ Data components imported")
        
        # Test main package
        import intelligent_automl
        print(f"✅ Main package imported, version: {intelligent_automl.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import pandas as pd
        from intelligent_automl.data import MissingValueHandler, DataPipeline
        
        # Create test data
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['x', None, 'z', 'w']
        })
        print("✅ Test data created")
        
        # Test missing value handler
        handler = MissingValueHandler()
        result = handler.fit_transform(df)
        print("✅ MissingValueHandler works")
        
        # Test pipeline
        pipeline = DataPipeline()
        pipeline.add_step('missing', MissingValueHandler())
        result = pipeline.fit_transform(df)
        print("✅ DataPipeline works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AutoML Framework Basic Test")
    print("=" * 40)
    
    success = True
    success &= test_basic_imports()
    success &= test_basic_functionality()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All basic tests passed!")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install framework: pip install -e .")
        print("3. Run full tests: python test_basic.py")
    else:
        print("❌ Some tests failed.")
