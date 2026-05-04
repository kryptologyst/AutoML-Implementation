"""Simple script to run the original AutoML code."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from automl import AutoMLTrainer, load_sample_data
except ImportError:
    # Fallback for direct execution
    import os
    os.chdir(Path(__file__).parent.parent)
    sys.path.insert(0, str(Path.cwd() / "src"))
    from automl import AutoMLTrainer, load_sample_data

def main():
    """Run the original AutoML implementation."""
    print("🤖 AutoML Implementation Demo")
    print("=" * 50)
    
    # Load sample data (equivalent to original digits dataset)
    print("Loading digits dataset...")
    X_train, X_test, y_train, y_test = load_sample_data("digits")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize AutoML trainer with TPOT
    print("\nInitializing TPOT AutoML model...")
    trainer = AutoMLTrainer(
        algorithm="tpot",
        random_state=42
    )
    
    # Fit the model
    print("Training AutoML model...")
    results = trainer.fit(X_train, y_train)
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"✅ AutoML Model Accuracy: {results['test_score']:.4f}")
    print(f"📈 Training Score: {results['train_score']:.4f}")
    print(f"🔄 CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print(f"⏱️  Training Time: {results['training_time']:.2f} seconds")
    
    # Additional information
    print(f"\n🔧 Algorithm: {results['algorithm']}")
    print(f"📊 Dataset: {results['dataset']}")
    
    print("\n" + "=" * 50)
    print("⚠️  Safety Notice:")
    print("This is a research demonstration. Results should not be")
    print("used for production decisions without proper validation.")
    print("=" * 50)
    
    print(f"\n👨‍💻 Author: kryptologyst")
    print(f"🔗 GitHub: https://github.com/kryptologyst")

if __name__ == "__main__":
    main()
