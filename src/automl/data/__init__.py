"""Data loading and preprocessing utilities for AutoML."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set CUDA/MPS seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")


def load_sample_data(
    dataset_name: str = "digits",
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessing: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess sample datasets.
    
    Args:
        dataset_name: Name of the dataset to load.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        preprocessing: Type of preprocessing to apply.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
        
    Raises:
        ValueError: If dataset_name is not supported.
    """
    set_random_seed(random_state)
    
    # Load dataset
    if dataset_name == "digits":
        data = load_digits()
    elif dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    elif dataset_name == "synthetic_classification":
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=random_state,
        )
        data = type("Dataset", (), {"data": X, "target": y})()
    elif dataset_name == "synthetic_regression":
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=0.1,
            random_state=random_state,
        )
        data = type("Dataset", (), {"data": X, "target": y})()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply preprocessing
    if preprocessing == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif preprocessing == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    logger.info(f"Loaded {dataset_name} dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test


def load_custom_data(
    file_path: Union[str, Path],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessing: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load custom dataset from file.
    
    Args:
        file_path: Path to the data file.
        target_column: Name of the target column.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        preprocessing: Type of preprocessing to apply.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    set_random_seed(random_state)
    
    file_path = Path(file_path)
    
    # Load data based on file extension
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Separate features and target
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply preprocessing
    if preprocessing == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif preprocessing == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    logger.info(f"Loaded custom dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test


def get_available_datasets() -> List[str]:
    """Get list of available sample datasets.
    
    Returns:
        List of available dataset names.
    """
    return [
        "digits",
        "iris", 
        "wine",
        "breast_cancer",
        "synthetic_classification",
        "synthetic_regression",
    ]


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Union[str, Path],
    dataset_name: str,
) -> None:
    """Save processed data to files.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training targets.
        y_test: Test targets.
        output_dir: Directory to save the data.
        dataset_name: Name of the dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.save(output_dir / f"{dataset_name}_X_train.npy", X_train)
    np.save(output_dir / f"{dataset_name}_X_test.npy", X_test)
    np.save(output_dir / f"{dataset_name}_y_train.npy", y_train)
    np.save(output_dir / f"{dataset_name}_y_test.npy", y_test)
    
    logger.info(f"Saved processed data to {output_dir}")


def load_processed_data(
    input_dir: Union[str, Path],
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load processed data from files.
    
    Args:
        input_dir: Directory containing the data files.
        dataset_name: Name of the dataset.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    input_dir = Path(input_dir)
    
    X_train = np.load(input_dir / f"{dataset_name}_X_train.npy")
    X_test = np.load(input_dir / f"{dataset_name}_X_test.npy")
    y_train = np.load(input_dir / f"{dataset_name}_y_train.npy")
    y_test = np.load(input_dir / f"{dataset_name}_y_test.npy")
    
    logger.info(f"Loaded processed data from {input_dir}")
    
    return X_train, X_test, y_train, y_test
