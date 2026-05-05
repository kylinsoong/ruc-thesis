import numpy as np
from typing import Dict, Any, List, Optional
import json
import os


def save_model(model: Any, filepath: str) -> None:
    import joblib
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    import joblib
    return joblib.load(filepath)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }
    
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            metrics["auc"] = 0.0
    
    return metrics


def save_results(results: Dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_experiment_results(
    model_name: str,
    metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "metrics": metrics,
        "config": config or {},
    }
