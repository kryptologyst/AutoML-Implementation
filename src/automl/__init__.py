"""AutoML Implementation Package.

A comprehensive AutoML (Automated Machine Learning) implementation featuring 
multiple algorithms, evaluation frameworks, and interactive demos for 
research and educational purposes.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from .data import load_sample_data, set_random_seed
from .models import create_model, get_available_models
from .train import AutoMLTrainer
from .eval import AutoMLEvaluator
from .utils import setup_logging, get_device_info

__all__ = [
    "load_sample_data",
    "set_random_seed", 
    "create_model",
    "get_available_models",
    "AutoMLTrainer",
    "AutoMLEvaluator",
    "setup_logging",
    "get_device_info",
]
