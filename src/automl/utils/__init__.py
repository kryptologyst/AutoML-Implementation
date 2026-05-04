"""Utility functions for AutoML."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.
    
    Args:
        log_file: Path to log file. If None, logs to console only.
        level: Logging level.
        format_string: Custom format string for log messages.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices.
    
    Returns:
        Dictionary containing device information.
    """
    device_info = {
        "cpu_count": os.cpu_count(),
        "platform": sys.platform,
        "python_version": sys.version,
    }
    
    # Check for CUDA
    try:
        import torch
        device_info["cuda_available"] = torch.cuda.is_available()
        if device_info["cuda_available"]:
            device_info["cuda_device_count"] = torch.cuda.device_count()
            device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        device_info["cuda_available"] = False
        
    # Check for MPS (Apple Silicon)
    try:
        import torch
        device_info["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        device_info["mps_available"] = False
        
    return device_info


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information.
    
    Returns:
        Dictionary containing memory usage in MB.
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {"rss": 0.0, "vms": 0.0, "percent": 0.0}


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_memory(memory_mb: float) -> str:
    """Format memory usage in a human-readable format.
    
    Args:
        memory_mb: Memory usage in MB.
        
    Returns:
        Formatted memory string.
    """
    if memory_mb < 1024:
        return f"{memory_mb:.2f} MB"
    elif memory_mb < 1024 * 1024:
        gb = memory_mb / 1024
        return f"{gb:.2f} GB"
    else:
        tb = memory_mb / (1024 * 1024)
        return f"{tb:.2f} TB"


def save_dict_to_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save.
        file_path: Path to save the file.
    """
    import yaml
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_dict_from_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from YAML file.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        Loaded dictionary.
    """
    import yaml
    
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """Create standard directory structure for AutoML project.
    
    Args:
        base_path: Base path for the project.
    """
    base_path = Path(base_path)
    
    directories = [
        "src/automl/data",
        "src/automl/models", 
        "src/automl/metrics",
        "src/automl/train",
        "src/automl/eval",
        "src/automl/viz",
        "src/automl/utils",
        "configs",
        "data/raw",
        "data/processed",
        "assets",
        "tests",
        "scripts",
        "demo",
        "notebooks",
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
        
    # Create .gitkeep files
    for directory in ["data/raw", "data/processed", "assets"]:
        (base_path / directory / ".gitkeep").touch()


def validate_data(
    X: np.ndarray,
    y: np.ndarray,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate input data for AutoML.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        task_type: Type of task ('classification', 'regression').
        
    Returns:
        Dictionary containing validation results.
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }
    
    # Check data types
    if not isinstance(X, np.ndarray):
        validation_results["errors"].append("X must be a numpy array")
        validation_results["is_valid"] = False
        
    if not isinstance(y, np.ndarray):
        validation_results["errors"].append("y must be a numpy array")
        validation_results["is_valid"] = False
        
    if not validation_results["is_valid"]:
        return validation_results
        
    # Check shapes
    if X.shape[0] != y.shape[0]:
        validation_results["errors"].append("X and y must have the same number of samples")
        validation_results["is_valid"] = False
        
    # Check for missing values
    if np.isnan(X).any():
        validation_results["warnings"].append("X contains NaN values")
        
    if np.isnan(y).any():
        validation_results["warnings"].append("y contains NaN values")
        
    # Check for infinite values
    if np.isinf(X).any():
        validation_results["warnings"].append("X contains infinite values")
        
    if np.isinf(y).any():
        validation_results["warnings"].append("y contains infinite values")
        
    # Auto-detect task type if not provided
    if task_type is None:
        unique_classes = len(np.unique(y))
        if unique_classes < 20:
            task_type = "classification"
        else:
            task_type = "regression"
            
    validation_results["info"]["task_type"] = task_type
    validation_results["info"]["n_samples"] = X.shape[0]
    validation_results["info"]["n_features"] = X.shape[1]
    validation_results["info"]["n_classes"] = len(np.unique(y)) if task_type == "classification" else None
    
    return validation_results


def get_model_summary(model: Any) -> Dict[str, Any]:
    """Get summary information about a model.
    
    Args:
        model: Trained model object.
        
    Returns:
        Dictionary containing model summary.
    """
    summary = {
        "model_type": type(model).__name__,
        "module": type(model).__module__,
    }
    
    # Try to get model parameters
    try:
        if hasattr(model, "get_params"):
            summary["parameters"] = model.get_params()
    except Exception:
        pass
        
    # Try to get feature importance
    try:
        if hasattr(model, "feature_importances_"):
            summary["feature_importances"] = model.feature_importances_.tolist()
    except Exception:
        pass
        
    # Try to get coefficients
    try:
        if hasattr(model, "coef_"):
            summary["coefficients"] = model.coef_.tolist()
    except Exception:
        pass
        
    return summary


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50,
) -> None:
    """Print a progress bar to console.
    
    Args:
        iteration: Current iteration.
        total: Total iterations.
        prefix: Prefix text.
        suffix: Suffix text.
        length: Length of the progress bar.
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    
    if iteration == total:
        print()
