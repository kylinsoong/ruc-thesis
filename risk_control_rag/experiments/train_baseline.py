"""
传统机器学习模型训练和评估脚本

功能：
- 支持命令行参数选择模型
- 在Credit Card Fraud和German Credit数据集上训练
- 计算评估指标：Accuracy、Precision、Recall、F1、AUC-ROC、AUC-PR
- 保存模型到 models/saved/
- 保存评估结果到 results/tables/baseline_results.csv

使用方法：
    python experiments/train_baseline.py --model logistic_regression --dataset credit_card
    python experiments/train_baseline.py --model random_forest --dataset german_credit
    python experiments/train_baseline.py --model xgboost --dataset all
    python experiments/train_baseline.py --model all --dataset all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import warnings

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from models.baseline.traditional_ml import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    create_model,
    get_available_models,
)
from config.model_config import get_model_config
from config.config import DATA_CONFIG, EXPERIMENT_CONFIG
from utils.data_loader import load_german_credit, load_credit_card_data, get_feature_info
from utils.preprocess import (
    preprocess_credit_card_data,
    preprocess_german_credit,
    get_dataset_statistics,
)

warnings.filterwarnings('ignore')


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（正类概率）
        
    Returns:
        Dict[str, float]: 评估指标字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = 0.0
    
    try:
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auc_pr"] = 0.0
    
    return metrics


def load_credit_card_dataset(base_dir: str) -> tuple:
    """加载Credit Card Fraud数据集"""
    processed_dir = Path(base_dir) / "processed"
    
    X_train_path = processed_dir / "X_train.csv"
    X_test_path = processed_dir / "X_test.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"
    
    if all(p.exists() for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("加载已处理的Credit Card数据集...")
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).squeeze()
        y_test = pd.read_csv(y_test_path).squeeze()
        return X_train, X_test, y_train, y_test
    
    raw_path = Path(base_dir) / "raw" / "creditcard.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Credit Card数据集不存在: {raw_path}\n"
            "请先下载数据集。"
        )
    
    print("预处理Credit Card数据集...")
    df = load_credit_card_data(str(raw_path))
    
    result = preprocess_credit_card_data(
        df,
        target_column='Class',
        standardize_cols=['Amount', 'Time'],
        apply_smote_oversampling=True,
        test_size=0.2,
        random_state=42,
    )
    
    return result["X_train"], result["X_test"], result["y_train"], result["y_test"]


def load_german_credit_dataset(base_dir: str) -> tuple:
    """加载German Credit数据集"""
    processed_dir = Path(base_dir) / "processed"
    
    X_train_path = processed_dir / "german_X_train.csv"
    X_test_path = processed_dir / "german_X_test.csv"
    y_train_path = processed_dir / "german_y_train.csv"
    y_test_path = processed_dir / "german_y_test.csv"
    
    if all(p.exists() for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("加载已处理的German Credit数据集...")
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).squeeze()
        y_test = pd.read_csv(y_test_path).squeeze()
        return X_train, X_test, y_train, y_test
    
    raw_path = Path(base_dir) / "raw" / "german.data"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"German Credit数据集不存在: {raw_path}\n"
            "请先下载数据集。"
        )
    
    print("预处理German Credit数据集...")
    df = load_german_credit(str(raw_path))
    
    feature_info = get_feature_info()
    
    X_train, X_test, y_train, y_test, preprocessors = preprocess_german_credit(
        df,
        categorical_cols=feature_info["categorical"],
        numerical_cols=feature_info["numerical"],
        target_col="class",
        encoding_method="label",
        test_size=0.2,
        random_state=42,
    )
    
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_frame().to_csv(y_train_path, index=False)
    y_test.to_frame().to_csv(y_test_path, index=False)
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(
    model_type: str,
    dataset_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> Dict[str, Any]:
    """
    训练和评估模型
    
    Args:
        model_type: 模型类型
        dataset_name: 数据集名称
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
        output_dir: 输出目录
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"训练模型: {model_type} | 数据集: {dataset_name}")
    print(f"{'='*60}")
    
    config = get_model_config(model_type, dataset_name)
    print(f"模型配置: {config}")
    
    model = create_model(model_type, **config)
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    model.train(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_all_metrics(y_test.values, y_pred, y_prob)
    
    print(f"\n评估指标:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    
    saved_dir = Path(output_dir) / "models" / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)
    model_path = saved_dir / f"{model_type}_{dataset_name}.joblib"
    model.save_model(str(model_path))
    
    feature_importance = model.get_feature_importance()
    if feature_importance:
        top_features = list(feature_importance.items())[:10]
        print(f"\nTop 10 重要特征:")
        for name, importance in top_features:
            print(f"  {name}: {importance:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(cm)
    
    return {
        "model_type": model_type,
        "dataset": dataset_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """保存结果到CSV文件"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    columns_order = [
        "model_type", "dataset", "timestamp",
        "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr",
        "train_samples", "test_samples"
    ]
    columns_order = [col for col in columns_order if col in df.columns]
    df = df[columns_order]
    
    if Path(output_path).exists():
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="训练和评估传统机器学习模型")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + get_available_models(),
        help="模型类型: logistic_regression, random_forest, xgboost, all"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "credit_card", "german_credit"],
        help="数据集: credit_card, german_credit, all"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据目录路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录路径"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    data_dir = args.data_dir or str(base_dir / "data")
    output_dir = args.output_dir or str(base_dir)
    
    model_types = get_available_models() if args.model == "all" else [args.model]
    dataset_names = ["credit_card", "german_credit"] if args.dataset == "all" else [args.dataset]
    
    all_results = []
    
    for dataset_name in dataset_names:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {dataset_name}")
        print(f"{'#'*60}")
        
        try:
            if dataset_name == "credit_card":
                X_train, X_test, y_train, y_test = load_credit_card_dataset(data_dir)
            else:
                X_train, X_test, y_train, y_test = load_german_credit_dataset(data_dir)
        except FileNotFoundError as e:
            print(f"跳过数据集 {dataset_name}: {e}")
            continue
        
        for model_type in model_types:
            try:
                result = train_and_evaluate_model(
                    model_type=model_type,
                    dataset_name=dataset_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    output_dir=output_dir,
                )
                all_results.append(result)
            except Exception as e:
                print(f"模型 {model_type} 在数据集 {dataset_name} 上训练失败: {e}")
                import traceback
                traceback.print_exc()
    
    if all_results:
        results_path = Path(output_dir) / "results" / "tables" / "baseline_results.csv"
        save_results_to_csv(all_results, str(results_path))
        
        print(f"\n{'='*60}")
        print("训练完成！结果汇总:")
        print(f"{'='*60}")
        
        results_df = pd.DataFrame(all_results)
        print(results_df[["model_type", "dataset", "accuracy", "f1", "auc_roc"]].to_string(index=False))
    else:
        print("\n没有成功训练的模型")


if __name__ == "__main__":
    main()
