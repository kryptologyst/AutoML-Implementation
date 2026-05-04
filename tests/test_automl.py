"""Test suite for AutoML implementation."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

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


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_sample_data(self):
        """Test loading sample datasets."""
        X_train, X_test, y_train, y_test = load_sample_data("digits")
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
    
    def test_set_random_seed(self):
        """Test random seed setting."""
        set_random_seed(42)
        # This should not raise an exception
        assert True


class TestModels:
    """Test AutoML model implementations."""
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model("random_forest")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "score")
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "random_forest" in models


class TestTraining:
    """Test training functionality."""
    
    def test_automl_trainer_init(self):
        """Test AutoML trainer initialization."""
        trainer = AutoMLTrainer(algorithm="random_forest")
        assert trainer.algorithm == "random_forest"
        assert trainer.random_state == 42
    
    def test_automl_trainer_fit(self):
        """Test AutoML trainer fitting."""
        trainer = AutoMLTrainer(algorithm="random_forest")
        
        # Use small dataset for testing
        X_train, X_test, y_train, y_test = load_sample_data("iris")
        
        results = trainer.fit(X_train, y_train)
        
        assert "train_score" in results
        assert "test_score" in results
        assert "cv_mean" in results
        assert "model" in results
        assert results["train_score"] > 0
        assert results["test_score"] > 0


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_automl_evaluator_init(self):
        """Test AutoML evaluator initialization."""
        evaluator = AutoMLEvaluator()
        assert evaluator.random_state == 42
    
    def test_evaluate_classification(self):
        """Test classification evaluation."""
        evaluator = AutoMLEvaluator()
        
        # Create dummy data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        
        metrics = evaluator.evaluate_classification(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert metrics["accuracy"] == 1.0
    
    def test_evaluate_regression(self):
        """Test regression evaluation."""
        evaluator = AutoMLEvaluator()
        
        # Create dummy data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluator.evaluate_regression(y_true, y_pred)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Load data
        X_train, X_test, y_train, y_test = load_sample_data("iris")
        
        # Train model
        trainer = AutoMLTrainer(algorithm="random_forest")
        results = trainer.fit(X_train, y_train)
        
        # Evaluate model
        evaluator = AutoMLEvaluator()
        evaluation = evaluator.evaluate_model(
            results["model"], X_test, y_test
        )
        
        assert "metrics" in evaluation
        assert "task_type" in evaluation
        assert evaluation["task_type"] == "classification"
    
    def test_multiple_algorithms(self):
        """Test multiple algorithm comparison."""
        X_train, X_test, y_train, y_test = load_sample_data("iris")
        
        trainer = AutoMLTrainer(algorithm="random_forest")
        results = trainer.compare_algorithms(
            X_train, y_train,
            algorithms=["random_forest", "svm"]
        )
        
        assert len(results) == 2
        assert "random_forest" in results
        assert "svm" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
