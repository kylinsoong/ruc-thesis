import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_prob, pd.Series):
        y_prob = y_prob.values

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = 0.0

    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auc_pr"] = 0.0

    return metrics


def calculate_metrics_with_std(
    results: list
) -> Dict[str, float]:
    metrics_names = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]
    result_dict = {}

    for name in metrics_names:
        values = [r[name] for r in results]
        result_dict[f"{name}_mean"] = float(np.mean(values))
        result_dict[f"{name}_std"] = float(np.std(values))

    return result_dict
