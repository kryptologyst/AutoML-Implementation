"""Simple test script to verify the basic structure works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that we can import the basic modules."""
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        print("✅ Basic imports successful")
        
        # Test data loading
        digits = load_digits()
        X, y = digits.data, digits.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"✅ Data loading successful: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # Test basic model training
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        print(f"✅ Basic model training successful: {accuracy:.4f} accuracy")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_automl_imports():
    """Test AutoML specific imports."""
    try:
        from automl.data import load_sample_data, set_random_seed
        from automl.models import create_model, get_available_models
        from automl.utils import setup_logging, get_device_info
        
        print("✅ AutoML imports successful")
        
        # Test basic functionality
        set_random_seed(42)
        models = get_available_models()
        print(f"✅ Available models: {len(models)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ AutoML import error: {e}")
        return False
    except Exception as e:
        print(f"❌ AutoML error: {e}")
        return False

def main():
    """Run basic tests."""
    print("🧪 Testing AutoML Implementation")
    print("=" * 50)
    
    # Test basic functionality
    basic_ok = test_imports()
    
    # Test AutoML functionality
    automl_ok = test_automl_imports()
    
    print("\n" + "=" * 50)
    if basic_ok and automl_ok:
        print("✅ All tests passed!")
        print("The AutoML implementation is ready to use.")
    else:
        print("❌ Some tests failed.")
        print("Please check the error messages above.")
    
    print("\n⚠️  Safety Notice:")
    print("This is a research demonstration. Results should not be")
    print("used for production decisions without proper validation.")
    
    print(f"\n👨‍💻 Author: kryptologyst")
    print(f"🔗 GitHub: https://github.com/kryptologyst")

if __name__ == "__main__":
    main()
