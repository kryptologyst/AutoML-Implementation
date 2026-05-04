"""Setup script for AutoML implementation."""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    try:
        # Test basic imports
        import numpy
        import pandas
        import sklearn
        print("✅ Basic packages imported successfully")
        
        # Test AutoML imports
        sys.path.append(str(Path(__file__).parent / "src"))
        from automl import load_sample_data, set_random_seed
        print("✅ AutoML package imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 AutoML Implementation Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        return False
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        return False
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run:")
    print("  python scripts/run_original.py")
    print("  python scripts/run_comparison.py")
    print("  streamlit run demo/streamlit_app.py")
    print("  python demo/gradio_app.py")
    
    print("\n⚠️  Safety Notice:")
    print("This is a research demonstration. Results should not be")
    print("used for production decisions without proper validation.")
    
    print(f"\n👨‍💻 Author: kryptologyst")
    print(f"🔗 GitHub: https://github.com/kryptologyst")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
