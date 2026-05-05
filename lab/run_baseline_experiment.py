import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
torch.set_num_threads(1)
from sklearn.model_selection import train_test_split
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from models.factory import create_model, get_available_models
from evaluation.metrics import calculate_metrics


RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    import random
    random.seed(seed)


def save_result(result: Dict, output_path: str):
    file_exists = Path(output_path).exists()

    with open(output_path, 'a', newline='') as f:
        fieldnames = ['model_type', 'dataset', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'train_samples', 'test_samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'model_type': result['model_type'],
            'dataset': result['dataset'],
            'accuracy': f"{result['accuracy']:.4f}",
            'precision': f"{result['precision']:.4f}",
            'recall': f"{result['recall']:.4f}",
            'f1': f"{result['f1']:.4f}",
            'auc_roc': f"{result['auc_roc']:.4f}",
            'auc_pr': f"{result['auc_pr']:.4f}",
            'train_samples': result['train_samples'],
            'test_samples': result['test_samples']
        })


def run_model_experiment(
    model_type: str,
    dataset_name: str,
    mock_llm: bool = False,
    n_repeats: int = 1,
    n_workers: int = 10
) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"Running {model_type} on {dataset_name}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = load_data(dataset_name)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train class distribution: {Counter(y_train)}")
    print(f"Test class distribution: {Counter(y_test)}")

    results = []

    for repeat in range(n_repeats):
        print(f"\n--- Repeat {repeat + 1}/{n_repeats} ---")
        set_random_seed(RANDOM_SEED + repeat)

        if model_type == "llm":
            model = create_model("llm", mock_mode=mock_llm, random_state=RANDOM_SEED, n_workers=n_workers)
        elif model_type == "xgboost":
            counter = Counter(y_train)
            scale_pos_weight = counter[0] / counter[1] if counter[1] > 0 else 1
            model = create_model("xgboost", scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED)
            model.train(X_train, y_train)
        elif model_type in ["deepfm", "tabnet"]:
            model = create_model(model_type, random_state=RANDOM_SEED)
            model.train(X_train, y_train)
        else:
            model = create_model(model_type, random_state=RANDOM_SEED)
            model.train(X_train, y_train)

        print(f"Evaluating...")
        metrics = model.evaluate(X_test, y_test)

        result = {
            'model_type': model_type,
            'dataset': dataset_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc_roc': metrics['auc_roc'],
            'auc_pr': metrics['auc_pr'],
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        results.append(result)

        if model_type == "llm" and not mock_llm:
            time.sleep(1)

    if n_repeats > 1:
        avg_result = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
        std_result = {k: np.std([r[k] for r in results]) for k in results[0].keys()}
        print(f"\n--- Average over {n_repeats} repeats ---")
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
            print(f"  {key}: {avg_result[key]:.4f} ± {std_result[key]:.4f}")

    return results


def run_all_experiments(
    mock_llm: bool = False,
    n_repeats: int = 1,
    output_path: str = None,
    n_workers: int = 10
):
    if output_path is None:
        output_path = Path(__file__).parent / "results" / "baseline_results.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = get_available_models()
    datasets = ["german_credit", "credit_card"]

    print(f"\nRunning all baseline experiments")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Output: {output_path}")

    for dataset in datasets:
        for model in models:
            try:
                results = run_model_experiment(model, dataset, mock_llm=mock_llm, n_repeats=n_repeats, n_workers=n_workers)

                for result in results:
                    save_result(result, str(output_path))

                print(f"\nResult saved to {output_path}")

            except Exception as e:
                print(f"\nError running {model} on {dataset}: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments for financial risk control")
    parser.add_argument("--model", type=str, default=None, help="Model type to run (lr, rf, xgboost, deepfm, tabnet, llm)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use (german_credit, credit_card)")
    parser.add_argument("--n-repeats", type=int, default=1, help="Number of experiment repeats")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM responses")
    parser.add_argument("--all-models", action="store_true", help="Run all models")
    parser.add_argument("--all-datasets", action="store_true", help="Run on all datasets")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--n-workers", type=int, default=10, help="Number of parallel workers for LLM model")

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).parent / "results" / "baseline_results.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.all_models:
        run_all_experiments(mock_llm=args.mock_llm, n_repeats=args.n_repeats, output_path=str(output_path), n_workers=args.n_workers)
        return

    model = args.model
    dataset = args.dataset

    if model is None and dataset is None:
        print("No specific model/dataset specified. Running all experiments...")
        run_all_experiments(mock_llm=args.mock_llm, n_repeats=args.n_repeats, output_path=str(output_path), n_workers=args.n_workers)
        return

    if model is None:
        print("Error: --model is required unless --all-models is specified")
        return

    if dataset is None:
        print("Error: --dataset is required unless --all-datasets is specified")
        return

    results = run_model_experiment(model, dataset, mock_llm=args.mock_llm, n_repeats=args.n_repeats, n_workers=args.n_workers)

    for result in results:
        save_result(result, str(output_path))

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
