"""Jupyter notebook for AutoML exploration."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from automl import (
    AutoMLTrainer,
    AutoMLEvaluator,
    load_sample_data,
    create_model,
    get_available_models,
    set_random_seed,
)

# Set random seed for reproducibility
set_random_seed(42)

print("🤖 AutoML Implementation - Interactive Notebook")
print("=" * 60)
print("Author: kryptologyst")
print("GitHub: https://github.com/kryptologyst")
print("=" * 60)

# Available datasets
print("\n📊 Available Datasets:")
datasets = ["digits", "iris", "wine", "breast_cancer", "synthetic_classification"]
for i, dataset in enumerate(datasets, 1):
    print(f"  {i}. {dataset}")

# Available models
print("\n🤖 Available Models:")
models = get_available_models()
for i, model in enumerate(models, 1):
    print(f"  {i}. {model}")

print("\n" + "=" * 60)
print("⚠️  Safety Notice:")
print("This is a research demonstration. Results should not be")
print("used for production decisions without proper validation.")
print("=" * 60)

# Example usage
print("\n💡 Example Usage:")
print("""
# Load data
X_train, X_test, y_train, y_test = load_sample_data("digits")

# Train a model
trainer = AutoMLTrainer(algorithm="tpot")
results = trainer.fit(X_train, y_train)

# Evaluate
evaluator = AutoMLEvaluator()
evaluation = evaluator.evaluate_model(results["model"], X_test, y_test)

print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
""")
