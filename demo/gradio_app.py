"""Gradio demo for AutoML implementation."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from automl import AutoMLTrainer, load_sample_data, get_available_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_automl_experiment(
    dataset: str,
    algorithms: List[str],
    generations: int,
    population_size: int,
    time_limit: int,
    n_trials: int,
    cv_folds: int,
    test_size: float,
) -> Tuple[str, str, str]:
    """Run AutoML experiment and return results.
    
    Args:
        dataset: Dataset name.
        algorithms: List of algorithms to run.
        generations: TPOT generations.
        population_size: TPOT population size.
        time_limit: Auto-Sklearn time limit.
        n_trials: Optuna number of trials.
        cv_folds: Cross-validation folds.
        test_size: Test set size.
        
    Returns:
        Tuple of (results_table, performance_chart, detailed_results).
    """
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_sample_data(
            dataset_name=dataset,
            test_size=test_size,
            random_state=42,
            preprocessing="standard"
        )
        
        results = {}
        training_times = {}
        
        # Train each algorithm
        for algorithm in algorithms:
            try:
                # Configure algorithm-specific parameters
                kwargs = {}
                
                if algorithm == "tpot":
                    kwargs = {
                        "generations": generations,
                        "population_size": population_size,
                    }
                elif algorithm == "auto_sklearn":
                    kwargs = {
                        "time_left_for_this_task": time_limit,
                    }
                elif algorithm == "optuna":
                    kwargs = {
                        "n_trials": n_trials,
                    }
                
                # Create trainer
                trainer = AutoMLTrainer(
                    algorithm=algorithm,
                    random_state=42,
                    **kwargs
                )
                
                # Train model
                import time
                start_time = time.time()
                result = trainer.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Store results
                results[algorithm] = result
                training_times[algorithm] = training_time
                
            except Exception as e:
                logger.error(f"Training failed for {algorithm}: {str(e)}")
                results[algorithm] = {"error": str(e)}
        
        # Create results table
        results_data = []
        for name, result in results.items():
            if "error" not in result:
                results_data.append({
                    "Algorithm": name,
                    "Train Score": f"{result['train_score']:.4f}",
                    "Test Score": f"{result['test_score']:.4f}",
                    "CV Score": f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                    "Training Time": f"{training_times.get(name, 0):.2f}s",
                })
        
        if not results_data:
            return "❌ No successful results", "", "❌ All algorithms failed"
        
        results_df = pd.DataFrame(results_data)
        results_table = results_df.to_string(index=False)
        
        # Create performance chart data
        algorithms = [row["Algorithm"] for row in results_data]
        test_scores = [float(row["Test Score"]) for row in results_data]
        times = [float(row["Training Time"].replace("s", "")) for row in results_data]
        
        chart_data = pd.DataFrame({
            "Algorithm": algorithms,
            "Test Score": test_scores,
            "Training Time": times
        })
        
        performance_chart = chart_data.to_string(index=False)
        
        # Detailed results
        detailed_results = "📊 Detailed Results:\n\n"
        for name, result in results.items():
            if "error" not in result:
                detailed_results += f"**{name}:**\n"
                detailed_results += f"  - Train Score: {result['train_score']:.4f}\n"
                detailed_results += f"  - Test Score: {result['test_score']:.4f}\n"
                detailed_results += f"  - CV Mean: {result['cv_mean']:.4f}\n"
                detailed_results += f"  - CV Std: {result['cv_std']:.4f}\n"
                detailed_results += f"  - Training Time: {training_times.get(name, 0):.2f}s\n\n"
        
        return results_table, performance_chart, detailed_results
        
    except Exception as e:
        error_msg = f"❌ Experiment failed: {str(e)}"
        return error_msg, "", error_msg


# Create Gradio interface
with gr.Blocks(
    title="AutoML Implementation Demo",
    theme=gr.themes.Soft(),
) as demo:
    
    gr.Markdown("""
    # 🤖 AutoML Implementation Demo
    
    A comprehensive demonstration of Automated Machine Learning algorithms including TPOT, 
    Auto-Sklearn, Optuna, and classical baselines for research and educational purposes.
    
    **⚠️ Safety Notice:** This is a research demonstration. Results should not be used for 
    production decisions without proper validation.
    
    **Author:** [kryptologyst](https://github.com/kryptologyst)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            
            dataset = gr.Dropdown(
                choices=[
                    ("Digits Classification", "digits"),
                    ("Iris Classification", "iris"),
                    ("Wine Classification", "wine"),
                    ("Breast Cancer Classification", "breast_cancer"),
                    ("Synthetic Classification", "synthetic_classification"),
                    ("Synthetic Regression", "synthetic_regression"),
                ],
                value="digits",
                label="Dataset"
            )
            
            algorithms = gr.CheckboxGroup(
                choices=[
                    ("TPOT", "tpot"),
                    ("Auto-Sklearn", "auto_sklearn"),
                    ("Optuna", "optuna"),
                    ("Random Forest", "random_forest"),
                    ("SVM", "svm"),
                    ("Logistic Regression", "logistic"),
                    ("Decision Tree", "decision_tree"),
                ],
                value=["tpot", "random_forest"],
                label="Algorithms"
            )
            
            gr.Markdown("### TPOT Parameters")
            generations = gr.Slider(1, 20, value=5, step=1, label="Generations")
            population_size = gr.Slider(10, 100, value=20, step=5, label="Population Size")
            
            gr.Markdown("### Auto-Sklearn Parameters")
            time_limit = gr.Slider(60, 600, value=300, step=30, label="Time Limit (seconds)")
            
            gr.Markdown("### Optuna Parameters")
            n_trials = gr.Slider(10, 200, value=100, step=10, label="Number of Trials")
            
            gr.Markdown("### General Parameters")
            cv_folds = gr.Slider(3, 10, value=5, step=1, label="CV Folds")
            test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Test Size")
            
            run_button = gr.Button("🚀 Run Experiment", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            
            results_table = gr.Textbox(
                label="Results Table",
                lines=10,
                interactive=False
            )
            
            performance_chart = gr.Textbox(
                label="Performance Chart",
                lines=5,
                interactive=False
            )
            
            detailed_results = gr.Textbox(
                label="Detailed Results",
                lines=15,
                interactive=False
            )
    
    # Event handlers
    run_button.click(
        fn=run_automl_experiment,
        inputs=[
            dataset,
            algorithms,
            generations,
            population_size,
            time_limit,
            n_trials,
            cv_folds,
            test_size,
        ],
        outputs=[results_table, performance_chart, detailed_results]
    )
    
    # Examples
    gr.Markdown("### 💡 Example Configurations")
    
    with gr.Row():
        gr.Examples(
            examples=[
                ["digits", ["tpot", "random_forest"], 5, 20, 300, 100, 5, 0.2],
                ["wine", ["auto_sklearn", "svm"], 5, 20, 180, 50, 5, 0.2],
                ["iris", ["optuna", "logistic"], 3, 10, 120, 30, 3, 0.3],
            ],
            inputs=[
                dataset,
                algorithms,
                generations,
                population_size,
                time_limit,
                n_trials,
                cv_folds,
                test_size,
            ],
            label="Quick Start Examples"
        )
    
    # Information
    gr.Markdown("""
    ### ℹ️ About This Demo
    
    This interactive demo showcases various AutoML algorithms and their performance on different datasets.
    
    **Available Algorithms:**
    - **TPOT**: Tree-based Pipeline Optimization Tool using genetic algorithms
    - **Auto-Sklearn**: Automated machine learning with scikit-learn  
    - **Optuna**: Hyperparameter optimization framework
    - **Classical Baselines**: Random Forest, SVM, Logistic Regression, Decision Tree
    
    **Features:**
    - Interactive algorithm comparison
    - Real-time performance visualization
    - Detailed evaluation metrics
    - Configurable parameters
    
    **Safety Notice:**
    This is a research demonstration. Results should not be used for production decisions without proper validation.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
