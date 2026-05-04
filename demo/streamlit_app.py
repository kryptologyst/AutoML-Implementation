"""Streamlit demo for AutoML implementation."""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from automl import AutoMLTrainer, AutoMLEvaluator, load_sample_data, get_available_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AutoML Implementation Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safety disclaimer
st.sidebar.markdown("""
## ⚠️ Safety & Ethics Disclaimer

**This is a research and educational demonstration. NOT for production decisions or control.**

- Results should not be used for critical decision-making without human oversight
- Always validate AutoML results with domain experts
- Consider bias, fairness, and interpretability in real-world applications
- Respect data privacy and licensing requirements
""")

# Title and description
st.title("🤖 AutoML Implementation Demo")
st.markdown("""
A comprehensive demonstration of Automated Machine Learning algorithms including TPOT, 
Auto-Sklearn, Optuna, and classical baselines for research and educational purposes.

**Author:** [kryptologyst](https://github.com/kryptologyst)
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Dataset selection
dataset_options = {
    "Digits Classification": "digits",
    "Iris Classification": "iris", 
    "Wine Classification": "wine",
    "Breast Cancer Classification": "breast_cancer",
    "Synthetic Classification": "synthetic_classification",
    "Synthetic Regression": "synthetic_regression",
}

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=list(dataset_options.keys()),
    index=0
)

dataset_name = dataset_options[selected_dataset]

# Algorithm selection
algorithm_options = {
    "TPOT": "tpot",
    "Auto-Sklearn": "auto_sklearn", 
    "Optuna": "optuna",
    "Random Forest": "random_forest",
    "SVM": "svm",
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
}

selected_algorithms = st.sidebar.multiselect(
    "Select Algorithms",
    options=list(algorithm_options.keys()),
    default=["TPOT", "Random Forest"]
)

# Training parameters
st.sidebar.subheader("Training Parameters")

if "TPOT" in selected_algorithms:
    st.sidebar.markdown("**TPOT Settings:**")
    tpot_generations = st.sidebar.slider("Generations", 1, 20, 5)
    tpot_population = st.sidebar.slider("Population Size", 10, 100, 20)

if "Auto-Sklearn" in selected_algorithms:
    st.sidebar.markdown("**Auto-Sklearn Settings:**")
    autosklearn_time = st.sidebar.slider("Time Limit (seconds)", 60, 600, 300)

if "Optuna" in selected_algorithms:
    st.sidebar.markdown("**Optuna Settings:**")
    optuna_trials = st.sidebar.slider("Number of Trials", 10, 200, 100)
    optuna_base_model = st.sidebar.selectbox(
        "Base Model", 
        ["random_forest", "svm", "logistic"]
    )

# Evaluation parameters
st.sidebar.subheader("Evaluation Parameters")
cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, 5)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Main content
if st.button("🚀 Run AutoML Experiment", type="primary"):
    
    if not selected_algorithms:
        st.error("Please select at least one algorithm!")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load data
    status_text.text("Loading data...")
    progress_bar.progress(10)
    
    try:
        X_train, X_test, y_train, y_test = load_sample_data(
            dataset_name=dataset_name,
            test_size=test_size,
            random_state=42,
            preprocessing="standard"
        )
        
        st.success(f"✅ Loaded {dataset_name} dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        st.stop()
    
    # Initialize results storage
    results = {}
    training_times = {}
    
    # Train models
    total_algorithms = len(selected_algorithms)
    
    for i, algorithm_name in enumerate(selected_algorithms):
        algorithm = algorithm_options[algorithm_name]
        
        status_text.text(f"Training {algorithm_name}...")
        progress = 20 + (i / total_algorithms) * 60
        progress_bar.progress(int(progress))
        
        try:
            # Configure algorithm-specific parameters
            kwargs = {}
            
            if algorithm == "tpot":
                kwargs = {
                    "generations": tpot_generations,
                    "population_size": tpot_population,
                }
            elif algorithm == "auto_sklearn":
                kwargs = {
                    "time_left_for_this_task": autosklearn_time,
                }
            elif algorithm == "optuna":
                kwargs = {
                    "n_trials": optuna_trials,
                    "base_model": optuna_base_model,
                }
            
            # Create trainer
            trainer = AutoMLTrainer(
                algorithm=algorithm,
                random_state=42,
                **kwargs
            )
            
            # Train model
            start_time = time.time()
            result = trainer.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Store results
            results[algorithm_name] = result
            training_times[algorithm_name] = training_time
            
            st.success(f"✅ {algorithm_name} completed in {training_time:.2f}s")
            
        except Exception as e:
            st.error(f"❌ {algorithm_name} failed: {str(e)}")
            logger.error(f"Training failed for {algorithm_name}: {str(e)}")
    
    progress_bar.progress(100)
    status_text.text("Evaluation complete!")
    
    # Display results
    if results:
        st.header("📊 Results")
        
        # Create results DataFrame
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
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Performance comparison chart
            st.subheader("📈 Performance Comparison")
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Test Score", "CV Score", "Training Time", "Score vs Time"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            algorithms = [row["Algorithm"] for row in results_data]
            test_scores = [float(row["Test Score"]) for row in results_data]
            cv_scores = [float(row["CV Score"].split(" ±")[0]) for row in results_data]
            times = [float(row["Training Time"].replace("s", "")) for row in results_data]
            
            # Test scores
            fig.add_trace(
                go.Bar(x=algorithms, y=test_scores, name="Test Score", marker_color="lightblue"),
                row=1, col=1
            )
            
            # CV scores
            fig.add_trace(
                go.Bar(x=algorithms, y=cv_scores, name="CV Score", marker_color="lightgreen"),
                row=1, col=2
            )
            
            # Training times
            fig.add_trace(
                go.Bar(x=algorithms, y=times, name="Training Time", marker_color="lightcoral"),
                row=2, col=1
            )
            
            # Score vs Time scatter
            fig.add_trace(
                go.Scatter(x=times, y=test_scores, mode="markers+text",
                          text=algorithms, textposition="top center",
                          name="Score vs Time", marker=dict(size=10, color="purple")),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False, title_text="AutoML Performance Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed evaluation
            st.subheader("🔍 Detailed Evaluation")
            
            evaluator = AutoMLEvaluator()
            
            for name, result in results.items():
                if "error" not in result:
                    with st.expander(f"📋 {name} Details"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Metrics:**")
                            st.json({
                                "Train Score": f"{result['train_score']:.4f}",
                                "Test Score": f"{result['test_score']:.4f}",
                                "CV Mean": f"{result['cv_mean']:.4f}",
                                "CV Std": f"{result['cv_std']:.4f}",
                                "Training Time": f"{training_times.get(name, 0):.2f}s",
                            })
                        
                        with col2:
                            st.markdown("**CV Scores:**")
                            cv_scores_list = result['cv_scores']
                            st.line_chart(pd.DataFrame({"CV Score": cv_scores_list}))
        
        # Download results
        st.subheader("💾 Download Results")
        
        if st.button("Download Results as CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"automl_results_{dataset_name}.csv",
                mime="text/csv"
            )

# Information section
st.header("ℹ️ About This Demo")

st.markdown("""
This interactive demo showcases various AutoML algorithms and their performance on different datasets.

### Available Algorithms:
- **TPOT**: Tree-based Pipeline Optimization Tool using genetic algorithms
- **Auto-Sklearn**: Automated machine learning with scikit-learn  
- **Optuna**: Hyperparameter optimization framework
- **Classical Baselines**: Random Forest, SVM, Logistic Regression, Decision Tree

### Features:
- Interactive algorithm comparison
- Real-time performance visualization
- Detailed evaluation metrics
- Downloadable results
- Configurable parameters

### Safety Notice:
This is a research demonstration. Results should not be used for production decisions without proper validation.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created by <a href='https://github.com/kryptologyst'>kryptologyst</a> | 
    AutoML Implementation Demo | Research & Educational Use Only</p>
</div>
""", unsafe_allow_html=True)
