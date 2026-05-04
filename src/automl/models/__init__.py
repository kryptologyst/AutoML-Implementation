"""AutoML model implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)


class BaseAutoMLModel(ABC):
    """Base class for AutoML models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the AutoML model.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseAutoMLModel":
        """Fit the model to the data.
        
        Args:
            X: Training features.
            y: Training targets.
            
        Returns:
            Self for method chaining.
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Predictions.
        """
        pass
        
    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score the model on test data.
        
        Args:
            X: Test features.
            y: Test targets.
            
        Returns:
            Score value.
        """
        pass


class TPOTModel(BaseAutoMLModel):
    """TPOT (Tree-based Pipeline Optimization Tool) implementation."""
    
    def __init__(
        self,
        generations: int = 5,
        population_size: int = 20,
        cv: int = 5,
        random_state: int = 42,
        verbosity: int = 2,
        **kwargs
    ):
        """Initialize TPOT model.
        
        Args:
            generations: Number of generations to run.
            population_size: Size of the population.
            cv: Number of cross-validation folds.
            random_state: Random seed.
            verbosity: Verbosity level.
            **kwargs: Additional arguments for TPOTClassifier.
        """
        super().__init__(random_state)
        self.generations = generations
        self.population_size = population_size
        self.cv = cv
        self.verbosity = verbosity
        self.kwargs = kwargs
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TPOTModel":
        """Fit TPOT model to the data."""
        try:
            from tpot import TPOTClassifier, TPOTRegressor
            
            # Determine if classification or regression
            if len(np.unique(y)) < 20:  # Classification
                self.model = TPOTClassifier(
                    generations=self.generations,
                    population_size=self.population_size,
                    cv=self.cv,
                    random_state=self.random_state,
                    verbosity=self.verbosity,
                    **self.kwargs
                )
            else:  # Regression
                self.model = TPOTRegressor(
                    generations=self.generations,
                    population_size=self.population_size,
                    cv=self.cv,
                    random_state=self.random_state,
                    verbosity=self.verbosity,
                    **self.kwargs
                )
            
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("TPOT model fitted successfully")
            
        except ImportError:
            logger.error("TPOT not installed. Please install with: pip install tpot")
            raise
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using TPOT model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score TPOT model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X, y)


class AutoSklearnModel(BaseAutoMLModel):
    """Auto-Sklearn implementation."""
    
    def __init__(
        self,
        time_left_for_this_task: int = 300,
        per_run_time_limit: int = 30,
        memory_limit: int = 3072,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize Auto-Sklearn model.
        
        Args:
            time_left_for_this_task: Time limit in seconds.
            per_run_time_limit: Time limit per run in seconds.
            memory_limit: Memory limit in MB.
            random_state: Random seed.
            **kwargs: Additional arguments for AutoSklearnClassifier/Regressor.
        """
        super().__init__(random_state)
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.memory_limit = memory_limit
        self.kwargs = kwargs
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoSklearnModel":
        """Fit Auto-Sklearn model to the data."""
        try:
            import autosklearn.classification
            import autosklearn.regression
            
            # Determine if classification or regression
            if len(np.unique(y)) < 20:  # Classification
                self.model = autosklearn.classification.AutoSklearnClassifier(
                    time_left_for_this_task=self.time_left_for_this_task,
                    per_run_time_limit=self.per_run_time_limit,
                    memory_limit=self.memory_limit,
                    random_state=self.random_state,
                    **self.kwargs
                )
            else:  # Regression
                self.model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=self.time_left_for_this_task,
                    per_run_time_limit=self.per_run_time_limit,
                    memory_limit=self.memory_limit,
                    random_state=self.random_state,
                    **self.kwargs
                )
            
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Auto-Sklearn model fitted successfully")
            
        except ImportError:
            logger.error("Auto-Sklearn not installed. Please install with: pip install auto-sklearn")
            raise
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Auto-Sklearn model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score Auto-Sklearn model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X, y)


class OptunaModel(BaseAutoMLModel):
    """Optuna-based hyperparameter optimization implementation."""
    
    def __init__(
        self,
        base_model: str = "random_forest",
        n_trials: int = 100,
        cv: int = 5,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize Optuna model.
        
        Args:
            base_model: Base model to optimize ('random_forest', 'svm', 'logistic').
            n_trials: Number of optimization trials.
            cv: Number of cross-validation folds.
            random_state: Random seed.
            **kwargs: Additional arguments for Optuna.
        """
        super().__init__(random_state)
        self.base_model = base_model
        self.n_trials = n_trials
        self.cv = cv
        self.kwargs = kwargs
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptunaModel":
        """Fit Optuna model to the data."""
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            
            def objective(trial):
                if self.base_model == "random_forest":
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 10, 200),
                        max_depth=trial.suggest_int("max_depth", 3, 20),
                        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                        random_state=self.random_state,
                    )
                elif self.base_model == "svm":
                    model = SVC(
                        C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                        gamma=trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
                        random_state=self.random_state,
                    )
                elif self.base_model == "logistic":
                    model = LogisticRegression(
                        C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                        max_iter=trial.suggest_int("max_iter", 100, 1000),
                        random_state=self.random_state,
                    )
                else:
                    raise ValueError(f"Unsupported base model: {self.base_model}")
                
                score = cross_val_score(model, X, y, cv=self.cv).mean()
                return score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
            
            # Train best model
            best_params = study.best_params
            if self.base_model == "random_forest":
                self.model = RandomForestClassifier(**best_params, random_state=self.random_state)
            elif self.base_model == "svm":
                self.model = SVC(**best_params, random_state=self.random_state)
            elif self.base_model == "logistic":
                self.model = LogisticRegression(**best_params, random_state=self.random_state)
            
            self.model.fit(X, y)
            self.is_fitted = True
            self.best_score = study.best_value
            logger.info(f"Optuna model fitted successfully with score: {self.best_score:.4f}")
            
        except ImportError:
            logger.error("Optuna not installed. Please install with: pip install optuna")
            raise
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Optuna model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score Optuna model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X, y)


class ClassicalBaselineModel(BaseAutoMLModel):
    """Classical baseline models for comparison."""
    
    def __init__(
        self,
        model_name: str = "random_forest",
        random_state: int = 42,
        **kwargs
    ):
        """Initialize classical baseline model.
        
        Args:
            model_name: Name of the model ('random_forest', 'svm', 'logistic', 'decision_tree').
            random_state: Random seed.
            **kwargs: Additional arguments for the model.
        """
        super().__init__(random_state)
        self.model_name = model_name
        self.kwargs = kwargs
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalBaselineModel":
        """Fit classical baseline model to the data."""
        # Determine if classification or regression
        is_classification = len(np.unique(y)) < 20
        
        if self.model_name == "random_forest":
            if is_classification:
                self.model = RandomForestClassifier(random_state=self.random_state, **self.kwargs)
            else:
                self.model = RandomForestRegressor(random_state=self.random_state, **self.kwargs)
        elif self.model_name == "svm":
            if is_classification:
                self.model = SVC(random_state=self.random_state, **self.kwargs)
            else:
                self.model = SVR(**self.kwargs)
        elif self.model_name == "logistic":
            if is_classification:
                self.model = LogisticRegression(random_state=self.random_state, **self.kwargs)
            else:
                self.model = LinearRegression(**self.kwargs)
        elif self.model_name == "decision_tree":
            if is_classification:
                self.model = DecisionTreeClassifier(random_state=self.random_state, **self.kwargs)
            else:
                self.model = DecisionTreeRegressor(random_state=self.random_state, **self.kwargs)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"{self.model_name} model fitted successfully")
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using classical baseline model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score classical baseline model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X, y)


def get_available_models() -> List[str]:
    """Get list of available AutoML models.
    
    Returns:
        List of available model names.
    """
    return [
        "tpot",
        "auto_sklearn", 
        "optuna",
        "random_forest",
        "svm",
        "logistic",
        "decision_tree",
    ]


def create_model(model_name: str, **kwargs) -> BaseAutoMLModel:
    """Create an AutoML model instance.
    
    Args:
        model_name: Name of the model to create.
        **kwargs: Additional arguments for the model.
        
    Returns:
        AutoML model instance.
        
    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name == "tpot":
        return TPOTModel(**kwargs)
    elif model_name == "auto_sklearn":
        return AutoSklearnModel(**kwargs)
    elif model_name == "optuna":
        return OptunaModel(**kwargs)
    elif model_name in ["random_forest", "svm", "logistic", "decision_tree"]:
        return ClassicalBaselineModel(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
