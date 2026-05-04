"""Script to run comprehensive AutoML comparison."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from automl import AutoMLTrainer, AutoMLEvaluator, load_sample_data

def main():
    """Run comprehensive AutoML comparison."""
    print("🤖 Comprehensive AutoML Comparison")
    print("=" * 60)
    
    # Load data
    print("Loading digits dataset...")
    X_train, X_test, y_train, y_test = load_sample_data("digits")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define algorithms to compare
    algorithms = [
        ("TPOT", "tpot"),
        ("Auto-Sklearn", "auto_sklearn"),
        ("Optuna", "optuna"),
        ("Random Forest", "random_forest"),
        ("SVM", "svm"),
        ("Logistic Regression", "logistic"),
    ]
    
    print(f"\nComparing {len(algorithms)} algorithms...")
    print("-" * 60)
    
    # Initialize trainer and evaluator
    trainer = AutoMLTrainer(random_state=42)
    evaluator = AutoMLEvaluator()
    
    # Run comparison
    results = trainer.compare_algorithms(
        X_train, y_train,
        algorithms=[alg[1] for alg in algorithms]
    )
    
    # Display results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 60)
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(results, metric="accuracy")
    
    print(f"{'Rank':<4} {'Algorithm':<20} {'Accuracy':<10} {'CV Score':<15} {'Time (s)':<10}")
    print("-" * 60)
    
    for entry in leaderboard:
        rank = entry["rank"]
        algorithm = entry["model"]
        accuracy = entry["metrics"]["accuracy"]
        cv_score = f"{entry['metrics']['cv_mean']:.4f} ± {entry['metrics']['cv_std']:.4f}"
        time_taken = entry["metrics"].get("training_time", 0)
        
        print(f"{rank:<4} {algorithm:<20} {accuracy:<10.4f} {cv_score:<15} {time_taken:<10.2f}")
    
    # Detailed results
    print("\n🔍 DETAILED RESULTS")
    print("=" * 60)
    
    for name, result in results.items():
        if "error" in result:
            print(f"\n❌ {name}: {result['error']}")
            continue
            
        print(f"\n✅ {name}:")
        print(f"  📈 Train Score: {result['train_score']:.4f}")
        print(f"  📊 Test Score: {result['test_score']:.4f}")
        print(f"  🔄 CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"  ⏱️  Training Time: {result['training_time']:.2f}s")
    
    # Generate report
    print("\n📋 GENERATING REPORT...")
    report = evaluator.generate_report(results)
    
    # Save report
    report_path = Path("assets") / "automl_comparison_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("⚠️  Safety Notice:")
    print("This is a research demonstration. Results should not be")
    print("used for production decisions without proper validation.")
    print("=" * 60)
    
    print(f"\n👨‍💻 Author: kryptologyst")
    print(f"🔗 GitHub: https://github.com/kryptologyst")

if __name__ == "__main__":
    main()
