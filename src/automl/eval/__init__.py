"""Evaluation metrics and frameworks for AutoML."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)

logger = logging.getLogger(__name__)


class AutoMLEvaluator:
    """Comprehensive evaluation framework for AutoML models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize AutoML evaluator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate classification model performance.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
        }
        
        # Add AUC if probabilities are provided
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multiclass classification
                    metrics["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                
        return metrics
        
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate regression model performance.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
        
        # Add MAPE if no zero values
        if not np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics["mape"] = mape
            
        return metrics
        
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "auto",
    ) -> Dict[str, Any]:
        """Evaluate a trained model comprehensively.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test targets.
            task_type: Type of task ('classification', 'regression', 'auto').
            
        Returns:
            Dictionary of evaluation results.
        """
        # Auto-detect task type
        if task_type == "auto":
            task_type = "classification" if len(np.unique(y_test)) < 20 else "regression"
            
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass
                
        # Evaluate based on task type
        if task_type == "classification":
            metrics = self.evaluate_classification(y_test, y_pred, y_proba)
            
            # Add detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics["classification_report"] = report
            
            # Add confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
        else:  # regression
            metrics = self.evaluate_regression(y_test, y_pred)
            
        # Add prediction statistics
        metrics["prediction_stats"] = {
            "mean_pred": float(np.mean(y_pred)),
            "std_pred": float(np.std(y_pred)),
            "min_pred": float(np.min(y_pred)),
            "max_pred": float(np.max(y_pred)),
        }
        
        logger.info(f"Model evaluation completed for {task_type} task")
        
        return {
            "task_type": task_type,
            "metrics": metrics,
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist() if y_proba is not None else None,
        }
        
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "auto",
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names and trained models.
            X_test: Test features.
            y_test: Test targets.
            task_type: Type of task.
            
        Returns:
            Dictionary of evaluation results for each model.
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            
            try:
                results[name] = self.evaluate_model(model, X_test, y_test, task_type)
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {str(e)}")
                results[name] = {"error": str(e)}
                
        return results
        
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = "accuracy",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """Create a leaderboard from evaluation results.
        
        Args:
            results: Evaluation results from compare_models.
            metric: Metric to rank by.
            ascending: Whether to sort in ascending order.
            
        Returns:
            Sorted list of model results.
        """
        leaderboard = []
        
        for name, result in results.items():
            if "error" in result:
                continue
                
            entry = {
                "model": name,
                "task_type": result["task_type"],
                "metrics": result["metrics"],
            }
            
            # Add primary metric for sorting
            if metric in result["metrics"]:
                entry["primary_metric"] = result["metrics"][metric]
            else:
                entry["primary_metric"] = 0.0
                
            leaderboard.append(entry)
            
        # Sort by primary metric
        leaderboard.sort(key=lambda x: x["primary_metric"], reverse=not ascending)
        
        # Add rank
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
            
        logger.info(f"Created leaderboard with {len(leaderboard)} models")
        
        return leaderboard
        
    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results.
            output_path: Path to save the report.
            
        Returns:
            Report text.
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AutoML Model Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("Summary:")
        report_lines.append(f"  Number of models evaluated: {len(results)}")
        
        # Task types
        task_types = set(result.get("task_type", "unknown") for result in results.values())
        report_lines.append(f"  Task types: {', '.join(task_types)}")
        report_lines.append("")
        
        # Individual model results
        for name, result in results.items():
            if "error" in result:
                report_lines.append(f"Model: {name}")
                report_lines.append(f"  Error: {result['error']}")
                report_lines.append("")
                continue
                
            report_lines.append(f"Model: {name}")
            report_lines.append(f"  Task Type: {result['task_type']}")
            
            metrics = result["metrics"]
            report_lines.append("  Metrics:")
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"    {metric_name}: {value:.4f}")
                elif metric_name == "classification_report":
                    report_lines.append(f"    {metric_name}:")
                    for class_name, class_metrics in value.items():
                        if isinstance(class_metrics, dict):
                            report_lines.append(f"      {class_name}:")
                            for sub_metric, sub_value in class_metrics.items():
                                if isinstance(sub_value, (int, float)):
                                    report_lines.append(f"        {sub_metric}: {sub_value:.4f}")
                elif metric_name == "confusion_matrix":
                    report_lines.append(f"    {metric_name}: {value}")
                    
            report_lines.append("")
            
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
            
        return report_text
