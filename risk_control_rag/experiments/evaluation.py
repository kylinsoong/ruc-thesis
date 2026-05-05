import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model_utils import load_results
from config.config import EXPERIMENT_CONFIG


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def plot_metrics_comparison(results: Dict[str, Dict[str, float]], save_path: str):
    df = pd.DataFrame(results).T
    df.plot(kind="bar", figsize=(12, 6))
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    pass
    return ""


def main():
    print("Starting evaluation...")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
