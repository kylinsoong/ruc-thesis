"""
深度学习模型训练脚本

支持以下模型：
- DeepFM
- TabNet

在两个数据集上训练：
- Credit Card Fraud数据集
- German Credit数据集
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

import importlib.util
spec = importlib.util.spec_from_file_location(
    "deep_learning",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "baseline", "deep_learning.py")
)
deep_learning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deep_learning)
DeepFMModel = deep_learning.DeepFMModel
TabNetModel = deep_learning.TabNetModel
create_deep_learning_model = deep_learning.create_deep_learning_model
from utils.model_utils import calculate_metrics, save_results
from utils.data_loader import load_german_credit, GERMAN_CREDIT_CATEGORICAL, GERMAN_CREDIT_NUMERICAL
from utils.preprocess import preprocess_german_credit, load_preprocessors
from sklearn.model_selection import train_test_split


def load_credit_card_data() -> tuple:
    """
    加载Credit Card Fraud数据集
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()
    
    print(f"Credit Card数据加载完成:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    
    return X_train, X_test, y_train, y_test


def load_german_data() -> tuple:
    """
    加载German Credit数据集
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    
    df = load_german_credit(str(raw_dir / "german.data"))
    
    X_train, X_test, y_train, y_test, preprocessors = preprocess_german_credit(
        df,
        categorical_cols=GERMAN_CREDIT_CATEGORICAL,
        numerical_cols=GERMAN_CREDIT_NUMERICAL,
        encoding_method="label",
        test_size=0.2,
        random_state=42,
    )
    
    print(f"German Credit数据加载完成:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    
    return X_train, X_test, y_train, y_test


def get_model_config(model_type: str, dataset_name: str) -> Dict[str, Any]:
    """
    获取模型配置
    
    Args:
        model_type: 模型类型
        dataset_name: 数据集名称
        
    Returns:
        Dict: 模型配置
    """
    if model_type == "deepfm":
        if dataset_name == "credit_card":
            return {
                "embedding_dim": 16,
                "hidden_dims": [256, 128, 64],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 256,
                "epochs": 50,
                "early_stopping_patience": 10,
                "use_batch_norm": True,
                "random_state": 42,
            }
        else:
            return {
                "embedding_dim": 8,
                "hidden_dims": [64, 32],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "early_stopping_patience": 15,
                "use_batch_norm": True,
                "random_state": 42,
            }
    elif model_type == "tabnet":
        if dataset_name == "credit_card":
            return {
                "n_d": 16,
                "n_a": 16,
                "n_steps": 5,
                "gamma": 1.5,
                "n_independent": 2,
                "n_shared": 2,
                "lambda_sparse": 1e-3,
                "momentum": 0.3,
                "clip_value": 2.0,
                "optimizer_params": {"lr": 2e-2},
                "batch_size": 1024,
                "virtual_batch_size": 128,
                "max_epochs": 100,
                "patience": 15,
                "random_state": 42,
            }
        else:
            return {
                "n_d": 8,
                "n_a": 8,
                "n_steps": 3,
                "gamma": 1.3,
                "n_independent": 2,
                "n_shared": 2,
                "lambda_sparse": 1e-3,
                "momentum": 0.3,
                "clip_value": 2.0,
                "optimizer_params": {"lr": 2e-2},
                "batch_size": 256,
                "virtual_batch_size": 64,
                "max_epochs": 200,
                "patience": 20,
                "random_state": 42,
            }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    model_type: str,
    dataset_name: str,
    save_model_flag: bool = True,
    save_results_flag: bool = True,
) -> Dict[str, Any]:
    """
    训练和评估模型
    
    Args:
        model_type: 模型类型 ('deepfm' 或 'tabnet')
        dataset_name: 数据集名称 ('credit_card' 或 'german')
        save_model_flag: 是否保存模型
        save_results_flag: 是否保存结果
        
    Returns:
        Dict: 训练结果
    """
    print(f"\n{'='*60}")
    print(f"训练 {model_type.upper()} 模型在 {dataset_name} 数据集上")
    print(f"{'='*60}\n")
    
    if dataset_name == "credit_card":
        X_train, X_test, y_train, y_test = load_credit_card_data()
    elif dataset_name == "german":
        X_train, X_test, y_train, y_test = load_german_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )
    
    print(f"数据划分:")
    print(f"  训练集: {X_train_split.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  特征数: {X_train.shape[1]}")
    
    model_config = get_model_config(model_type, dataset_name)
    print(f"\n模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n开始训练...")
    model = create_deep_learning_model(model_type, **model_config)
    
    start_time = datetime.now()
    model.fit(X_train_split, y_train_split, X_val, y_val)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n训练完成! 耗时: {training_time:.2f} 秒")
    
    print(f"\n在测试集上评估...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    metrics = calculate_metrics(
        y_test.values if isinstance(y_test, pd.Series) else y_test,
        y_pred,
        y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob,
    )
    
    print(f"\n评估结果:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    results = {
        "model_type": model_type,
        "dataset": dataset_name,
        "model_config": model_config,
        "metrics": metrics,
        "training_time_seconds": training_time,
        "data_info": {
            "n_features": int(X_train.shape[1]),
            "n_train": int(X_train_split.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    if save_model_flag:
        model_dir = Path(__file__).parent.parent / "results" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_type}_{dataset_name}.pt"
        model.save(str(model_path))
    
    if save_results_flag:
        results_dir = Path(__file__).parent.parent / "results" / "tables"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / f"{model_type}_{dataset_name}_results.json"
        save_results(results, str(results_path))
    
    return results


def run_all_experiments():
    """
    运行所有实验
    """
    print("\n" + "="*60)
    print("开始运行所有深度学习模型实验")
    print("="*60)
    
    all_results = {}
    
    experiments = [
        ("deepfm", "credit_card"),
        ("deepfm", "german"),
        ("tabnet", "credit_card"),
        ("tabnet", "german"),
    ]
    
    for model_type, dataset_name in experiments:
        try:
            results = train_and_evaluate(model_type, dataset_name)
            all_results[f"{model_type}_{dataset_name}"] = results
        except Exception as e:
            print(f"\n错误: {model_type} 在 {dataset_name} 上训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("所有实验完成!")
    print("="*60)
    
    print("\n结果汇总:")
    print("-" * 80)
    print(f"{'模型':<15} {'数据集':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 80)
    
    for exp_name, results in all_results.items():
        metrics = results["metrics"]
        print(f"{results['model_type']:<15} {results['dataset']:<15} "
              f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} "
              f"{metrics.get('auc', 0):<10.4f}")
    
    summary_path = Path(__file__).parent.parent / "results" / "tables" / "deep_learning_summary.json"
    save_results(all_results, str(summary_path))
    print(f"\n结果已保存至: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="深度学习模型训练脚本")
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepfm", "tabnet", "all"],
        default="all",
        help="选择要训练的模型 (deepfm/tabnet/all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["credit_card", "german", "all"],
        default="all",
        help="选择数据集 (credit_card/german/all)",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="不保存模型",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="不保存结果",
    )
    
    args = parser.parse_args()
    
    if args.model == "all" and args.dataset == "all":
        run_all_experiments()
    else:
        if args.model == "all":
            models = ["deepfm", "tabnet"]
        else:
            models = [args.model]
        
        if args.dataset == "all":
            datasets = ["credit_card", "german"]
        else:
            datasets = [args.dataset]
        
        for model_type in models:
            for dataset_name in datasets:
                train_and_evaluate(
                    model_type,
                    dataset_name,
                    save_model_flag=not args.no_save_model,
                    save_results_flag=not args.no_save_results,
                )


if __name__ == "__main__":
    main()
