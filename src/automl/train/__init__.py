"""Training pipeline for AutoML models."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from sklearn.model_selection import cross_val_score

from ..data import load_sample_data, set_random_seed
from ..models import BaseAutoMLModel, create_model
from ..utils import setup_logging

logger = logging.getLogger(__name__)


class AutoMLTrainer:
    """AutoML training pipeline."""
    
    def __init__(
        self,
        algorithm: str = "tpot",
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "assets",
        random_state: int = 42,
        **kwargs
    ):
        """Initialize AutoML trainer.
        
        Args:
            algorithm: AutoML algorithm to use.
            config_path: Path to configuration file.
            output_dir: Directory to save outputs.
            random_state: Random seed for reproducibility.
            **kwargs: Additional arguments for the model.
        """
        self.algorithm = algorithm
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        setup_logging(self.output_dir / "training.log")
        
        # Set random seed
        set_random_seed(self.random_state)
        
        logger.info(f"Initialized AutoML trainer with algorithm: {algorithm}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary.
        """
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            config = self._get_default_config()
            logger.info("Using default configuration")
            
        return config
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Default configuration dictionary.
        """
        return {
            "experiment": {
                "name": f"{self.algorithm}_experiment",
                "algorithm": self.algorithm,
                "random_state": self.random_state,
            },
            "data": {
                "dataset": "digits",
                "test_size": 0.2,
                "preprocessing": "standard",
            },
            "model": {
                "generations": 5,
                "population_size": 20,
                "cv_folds": 5,
                "time_limit": 300,
            },
            "evaluation": {
                "metrics": ["accuracy", "f1_macro"],
                "cv_folds": 5,
                "save_artifacts": True,
            },
        }
        
    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fit AutoML model to data.
        
        Args:
            X: Training features. If None, will load sample data.
            y: Training targets. If None, will load sample data.
            dataset_name: Name of dataset to load if X, y are None.
            
        Returns:
            Dictionary containing training results.
        """
        # Load data if not provided
        if X is None or y is None:
            dataset_name = dataset_name or self.config["data"]["dataset"]
            X_train, X_test, y_train, y_test = load_sample_data(
                dataset_name=dataset_name,
                test_size=self.config["data"]["test_size"],
                random_state=self.random_state,
                preprocessing=self.config["data"]["preprocessing"],
            )
        else:
            # Split provided data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["data"]["test_size"],
                random_state=self.random_state,
                stratify=y if len(np.unique(y)) < 20 else None
            )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Create model
        model_kwargs = {**self.kwargs, **self.config["model"]}
        model = create_model(self.algorithm, **model_kwargs)
        
        # Train model
        logger.info(f"Starting training with {self.algorithm}")
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model.model if hasattr(model, 'model') else model,
                X_train, y_train,
                cv=self.config["evaluation"]["cv_folds"]
            )
            
            results = {
                "algorithm": self.algorithm,
                "dataset": dataset_name or "custom",
                "training_time": training_time,
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "cv_scores": cv_scores.tolist(),
                "model": model,
                "config": self.config,
            }
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Train score: {train_score:.4f}")
            logger.info(f"Test score: {test_score:.4f}")
            logger.info(f"CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Save results
            if self.config["evaluation"]["save_artifacts"]:
                self._save_results(results, X_test, y_test)
                
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def _save_results(
        self,
        results: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Save training results and artifacts.
        
        Args:
            results: Training results dictionary.
            X_test: Test features.
            y_test: Test targets.
        """
        import joblib
        import json
        
        # Save model
        model_path = self.output_dir / f"{self.algorithm}_model.joblib"
        joblib.dump(results["model"], model_path)
        
        # Save results (without model object)
        results_to_save = {k: v for k, v in results.items() if k != "model"}
        results_path = self.output_dir / f"{self.algorithm}_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
            
        # Save predictions
        predictions = results["model"].predict(X_test)
        np.save(self.output_dir / f"{self.algorithm}_predictions.npy", predictions)
        
        # Save configuration
        config_path = self.output_dir / f"{self.algorithm}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(results["config"], f, default_flow_style=False)
            
        logger.info(f"Results saved to {self.output_dir}")
        
    def compare_algorithms(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithms: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple AutoML algorithms.
        
        Args:
            X: Training features.
            y: Training targets.
            algorithms: List of algorithms to compare.
            **kwargs: Additional arguments for models.
            
        Returns:
            Dictionary containing results for each algorithm.
        """
        if algorithms is None:
            algorithms = ["tpot", "auto_sklearn", "optuna", "random_forest"]
            
        results = {}
        
        for algorithm in algorithms:
            logger.info(f"Comparing algorithm: {algorithm}")
            
            try:
                # Create trainer for this algorithm
                trainer = AutoMLTrainer(
                    algorithm=algorithm,
                    config_path=self.config_path,
                    output_dir=self.output_dir / algorithm,
                    random_state=self.random_state,
                    **kwargs
                )
                
                # Train and get results
                result = trainer.fit(X, y)
                results[algorithm] = result
                
            except Exception as e:
                logger.error(f"Failed to train {algorithm}: {str(e)}")
                results[algorithm] = {"error": str(e)}
                
        return results
        
    def load_model(self, model_path: Union[str, Path]) -> BaseAutoMLModel:
        """Load a trained model from file.
        
        Args:
            model_path: Path to the saved model.
            
        Returns:
            Loaded model.
        """
        import joblib
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        return model
        
    def predict(
        self,
        model: BaseAutoMLModel,
        X: np.ndarray,
    ) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            model: Trained model.
            X: Features to predict on.
            
        Returns:
            Predictions.
        """
        return model.predict(X)
