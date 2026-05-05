import numpy as np
from typing import List, Dict, Any
from scipy import stats


def paired_t_test(results1: List[Dict[str, float]], results2: List[Dict[str, float]], metric: str = "f1") -> float:
    values1 = [r[metric] for r in results1]
    values2 = [r[metric] for r in results2]

    if len(values1) != len(values2):
        raise ValueError("Results lists must have the same length")

    if len(values1) < 2:
        return 1.0

    t_statistic, p_value = stats.ttest_rel(values1, values2)
    return float(p_value)


def multiple_comparison_with_lr(
    lr_results: List[Dict[str, float]],
    other_results: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]
    comparison_results = {}

    for model_name, results in other_results.items():
        comparison_results[model_name] = {}
        for metric in metrics:
            try:
                p_value = paired_t_test(lr_results, results, metric)
                comparison_results[model_name][metric] = p_value
            except ValueError:
                comparison_results[model_name][metric] = 1.0

    return comparison_results
