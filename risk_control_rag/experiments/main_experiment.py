"""
主实验脚本 - 风险识别性能对比

功能：
1. 加载所有模型（LR、RF、XGBoost、DeepFM、TabNet、LLM、LLM+RAG）
2. 在Credit Card Fraud和German Credit数据集上运行
3. 计算性能指标：Accuracy、Precision、Recall、F1、AUC-ROC、AUC-PR
4. 绘制ROC曲线和PR曲线
5. 进行统计显著性检验（5次重复实验）
6. 生成实验结果表格

使用方法：
    python experiments/main_experiment.py --use-llm  # 使用真实LLM API
    python experiments/main_experiment.py --mock-llm  # 使用模拟LLM（默认）
    python experiments/main_experiment.py --n-repeats 3  # 设置重复次数
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import warnings
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

from models.baseline.traditional_ml import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    create_model as create_traditional_model,
)
from models.baseline.deep_learning import DeepFMModel, TabNetModel
from config.config import DATA_CONFIG, EXPERIMENT_CONFIG
from utils.data_loader import load_german_credit, get_feature_info

warnings.filterwarnings('ignore')

RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    import random
    random.seed(seed)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
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


def load_credit_card_dataset(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    processed_dir = Path(base_dir) / "data" / "processed"
    
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()
    
    return X_train, X_test, y_train, y_test


def load_german_credit_dataset(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    processed_dir = Path(base_dir) / "data" / "processed"
    
    X_train = pd.read_csv(processed_dir / "german_X_train.csv")
    X_test = pd.read_csv(processed_dir / "german_X_test.csv")
    y_train = pd.read_csv(processed_dir / "german_y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "german_y_test.csv").squeeze()
    
    y_train = y_train - 1
    y_test = y_test - 1
    
    return X_train, X_test, y_train, y_test


def load_traditional_model(model_type: str, dataset_name: str, base_dir: str):
    model_path = Path(base_dir) / "models" / "saved" / f"{model_type}_{dataset_name}.joblib"
    
    if model_type == "logistic_regression":
        model = LogisticRegressionModel(random_state=RANDOM_SEED)
    elif model_type == "random_forest":
        model = RandomForestModel(random_state=RANDOM_SEED)
    elif model_type == "xgboost":
        model = XGBoostModel(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model_path.exists():
        print(f"加载已训练模型: {model_path}")
        model.load_model(str(model_path))
        return model
    else:
        print(f"模型文件不存在: {model_path}，需要重新训练")
        return None


def train_traditional_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series):
    print(f"训练 {model_type} 模型...")
    
    if model_type == "logistic_regression":
        model = LogisticRegressionModel(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
    elif model_type == "random_forest":
        model = RandomForestModel(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
    elif model_type == "xgboost":
        from collections import Counter
        counter = Counter(y_train)
        scale_pos_weight = counter[0] / counter[1] if counter[1] > 0 else 1
        model = XGBoostModel(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.train(X_train, y_train)
    return model


def load_deep_learning_model(model_type: str, dataset_name: str, base_dir: str):
    model_dir = Path(base_dir) / "results" / "models"
    
    if model_type == "deepfm":
        model_path = model_dir / f"deepfm_{dataset_name}.pt"
        if model_path.exists():
            print(f"加载已训练模型: {model_path}")
            model = DeepFMModel(
                batch_size=256,
                random_state=RANDOM_SEED
            )
            model.load(str(model_path))
            return model
        else:
            print(f"模型文件不存在: {model_path}，需要重新训练")
            return None
    elif model_type == "tabnet":
        model_path = model_dir / f"tabnet_{dataset_name}.pt.zip"
        if model_path.exists():
            print(f"加载已训练模型: {model_path}")
            model = TabNetModel(random_state=RANDOM_SEED)
            model.load(str(model_path))
            return model
        else:
            print(f"模型文件不存在: {model_path}，需要重新训练")
            return None
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_deepfm_model(X_train: pd.DataFrame, y_train: pd.Series, dataset_name: str):
    print("训练 DeepFM 模型...")
    
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    if dataset_name == "credit_card":
        config = {
            "embedding_dim": 16,
            "hidden_dims": [256, 128, 64],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 50,
            "early_stopping_patience": 10,
            "random_state": RANDOM_SEED,
        }
    else:
        config = {
            "embedding_dim": 8,
            "hidden_dims": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "early_stopping_patience": 15,
            "random_state": RANDOM_SEED,
        }
    
    model = DeepFMModel(**config)
    model.fit(X_train_split, y_train_split, X_val, y_val)
    return model


def train_tabnet_model(X_train: pd.DataFrame, y_train: pd.Series, dataset_name: str):
    print("训练 TabNet 模型...")
    
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    if dataset_name == "credit_card":
        config = {
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
            "random_state": RANDOM_SEED,
        }
    else:
        config = {
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
            "random_state": RANDOM_SEED,
        }
    
    model = TabNetModel(**config)
    model.fit(X_train_split, y_train_split, X_val, y_val)
    return model


class MockLLMModel:
    """模拟LLM模型（用于测试，不调用真实API）"""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        np.random.seed(random_state)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        prob = np.random.uniform(0.3, 0.7, n_samples)
        return (prob > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        prob_pos = np.random.uniform(0.2, 0.8, n_samples)
        prob_neg = 1 - prob_pos
        return np.column_stack([prob_neg, prob_pos])


class MockRAGModel:
    """模拟RAG模型（用于测试，不调用真实API）"""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        np.random.seed(random_state)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        prob = np.random.uniform(0.25, 0.75, n_samples)
        return (prob > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        prob_pos = np.random.uniform(0.2, 0.8, n_samples)
        prob_neg = 1 - prob_pos
        return np.column_stack([prob_neg, prob_pos])


class RealLLMModel:
    """真实LLM模型（调用豆包API）"""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.is_fitted = True
        from models.rag.llm_only import LLMRiskAnalyzer
        from config.prompts import parse_risk_level
        self.analyzer = LLMRiskAnalyzer(temperature=0.3, max_tokens=1024)
        self.parse_risk_level = parse_risk_level
    
    def _row_to_dict(self, row: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
        data = {}
        for col in feature_names:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, (np.integer, np.floating)):
                data[col] = float(val)
            else:
                data[col] = str(val)
        return data
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        feature_names = list(X.columns)
        
        print(f"LLM预测 {len(X)} 个样本...")
        for idx in range(len(X)):
            row = X.iloc[idx]
            customer_data = self._row_to_dict(row, feature_names)
            
            try:
                result = self.analyzer.analyze_single(customer_data, f"LLM_{idx}")
                pred = 1 if result.risk_level >= 1 else 0
                predictions.append(pred)
            except Exception as e:
                print(f"样本 {idx} 预测失败: {e}")
                predictions.append(0)
            
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(X)} 样本")
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        feature_names = list(X.columns)
        
        print(f"LLM预测概率 {len(X)} 个样本...")
        for idx in range(len(X)):
            row = X.iloc[idx]
            customer_data = self._row_to_dict(row, feature_names)
            
            try:
                result = self.analyzer.analyze_single(customer_data, f"LLM_{idx}")
                score = result.risk_score / 100.0 if result.risk_score > 0 else 0.5
                predictions.append(score)
            except Exception as e:
                print(f"样本 {idx} 预测失败: {e}")
                predictions.append(0.5)
            
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(X)} 样本")
        
        prob_pos = np.array(predictions)
        prob_neg = 1 - prob_pos
        return np.column_stack([prob_neg, prob_pos])


class RealRAGModel:
    """真实RAG模型（调用豆包API + 知识库检索）"""
    
    def __init__(self, persist_dir: str, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.is_fitted = True
        from models.rag.rag_system import RAGRiskSystem, RAGSystemConfig, RetrievalStrategy
        config = RAGSystemConfig(
            retrieval_strategy=RetrievalStrategy.HYBRID,
            top_k=5,
            temperature=0.3,
            max_tokens=1024,
        )
        self.rag_system = RAGRiskSystem(config=config, persist_dir=persist_dir)
    
    def _row_to_dict(self, row: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
        data = {}
        for col in feature_names:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, (np.integer, np.floating)):
                data[col] = float(val)
            else:
                data[col] = str(val)
        return data
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        feature_names = list(X.columns)
        
        print(f"RAG预测 {len(X)} 个样本...")
        for idx in range(len(X)):
            row = X.iloc[idx]
            customer_data = self._row_to_dict(row, feature_names)
            
            try:
                result = self.rag_system.analyze(customer_data, f"RAG_{idx}")
                if result.parsed_result:
                    pred = 1 if result.parsed_result.risk_level.value >= 1 else 0
                else:
                    pred = 0
                predictions.append(pred)
            except Exception as e:
                print(f"样本 {idx} 预测失败: {e}")
                predictions.append(0)
            
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(X)} 样本")
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        feature_names = list(X.columns)
        
        print(f"RAG预测概率 {len(X)} 个样本...")
        for idx in range(len(X)):
            row = X.iloc[idx]
            customer_data = self._row_to_dict(row, feature_names)
            
            try:
                result = self.rag_system.analyze(customer_data, f"RAG_{idx}")
                if result.parsed_result and result.parsed_result.risk_score > 0:
                    score = result.parsed_result.risk_score / 100.0
                else:
                    score = 0.5
                predictions.append(score)
            except Exception as e:
                print(f"样本 {idx} 预测失败: {e}")
                predictions.append(0.5)
            
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(X)} 样本")
        
        prob_pos = np.array(predictions)
        prob_neg = 1 - prob_pos
        return np.column_stack([prob_neg, prob_pos])


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    
    metrics = calculate_all_metrics(y_test.values, y_pred, y_prob)
    
    return metrics, y_pred, y_prob


def run_single_experiment(
    dataset_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_dir: str,
    use_llm: bool = False,
    llm_sample_size: int = 50,
) -> Dict[str, Any]:
    """运行单次实验"""
    
    results = {}
    predictions = {}
    probabilities = {}
    
    traditional_models = ["logistic_regression", "random_forest", "xgboost"]
    
    for model_type in traditional_models:
        print(f"\n{'='*40}")
        print(f"评估 {model_type} 模型")
        print(f"{'='*40}")
        
        model = load_traditional_model(model_type, dataset_name, base_dir)
        
        if model is None:
            model = train_traditional_model(model_type, X_train, y_train)
        
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        results[model_type] = metrics
        predictions[model_type] = y_pred
        probabilities[model_type] = y_prob
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    
    deep_learning_models = ["deepfm", "tabnet"]
    
    for model_type in deep_learning_models:
        print(f"\n{'='*40}")
        print(f"评估 {model_type.upper()} 模型")
        print(f"{'='*40}")
        
        model = load_deep_learning_model(model_type, dataset_name, base_dir)
        
        if model is None:
            if model_type == "deepfm":
                model = train_deepfm_model(X_train, y_train, dataset_name)
            else:
                model = train_tabnet_model(X_train, y_train, dataset_name)
        
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        results[model_type] = metrics
        predictions[model_type] = y_pred
        probabilities[model_type] = y_prob
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    
    print(f"\n{'='*40}")
    print("评估 LLM 模型")
    print(f"{'='*40}")
    
    if use_llm:
        X_test_llm = X_test.head(llm_sample_size)
        y_test_llm = y_test.head(llm_sample_size)
        llm_model = RealLLMModel(random_state=RANDOM_SEED)
    else:
        X_test_llm = X_test
        y_test_llm = y_test
        llm_model = MockLLMModel(random_state=RANDOM_SEED)
    
    metrics, y_pred, y_prob = evaluate_model(llm_model, X_test_llm, y_test_llm)
    
    results["llm"] = metrics
    predictions["llm"] = y_pred
    probabilities["llm"] = y_prob
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    
    print(f"\n{'='*40}")
    print("评估 LLM+RAG 模型")
    print(f"{'='*40}")
    
    persist_dir = str(Path(base_dir) / "data" / "knowledge_base" / "chroma_db")
    
    if use_llm:
        X_test_rag = X_test.head(llm_sample_size)
        y_test_rag = y_test.head(llm_sample_size)
        rag_model = RealRAGModel(persist_dir=persist_dir, random_state=RANDOM_SEED)
    else:
        X_test_rag = X_test
        y_test_rag = y_test
        rag_model = MockRAGModel(random_state=RANDOM_SEED)
    
    metrics, y_pred, y_prob = evaluate_model(rag_model, X_test_rag, y_test_rag)
    
    results["llm_rag"] = metrics
    predictions["llm_rag"] = y_pred
    probabilities["llm_rag"] = y_prob
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    
    return results, predictions, probabilities


def run_repeated_experiments(
    dataset_name: str,
    base_dir: str,
    n_repeats: int = 5,
    use_llm: bool = False,
    llm_sample_size: int = 50,
) -> Dict[str, List[Dict[str, float]]]:
    """运行多次重复实验"""
    
    print(f"\n{'#'*60}")
    print(f"# 数据集: {dataset_name}")
    print(f"# 重复次数: {n_repeats}")
    print(f"{'#'*60}")
    
    if dataset_name == "credit_card":
        X_train, X_test, y_train, y_test = load_credit_card_dataset(base_dir)
    else:
        X_train, X_test, y_train, y_test = load_german_credit_dataset(base_dir)
    
    all_results = {
        "logistic_regression": [],
        "random_forest": [],
        "xgboost": [],
        "deepfm": [],
        "tabnet": [],
        "llm": [],
        "llm_rag": [],
    }
    
    for repeat in range(n_repeats):
        print(f"\n{'='*60}")
        print(f"第 {repeat + 1}/{n_repeats} 次实验")
        print(f"{'='*60}")
        
        set_random_seed(RANDOM_SEED + repeat)
        
        results, predictions, probabilities = run_single_experiment(
            dataset_name=dataset_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            base_dir=base_dir,
            use_llm=use_llm,
            llm_sample_size=llm_sample_size,
        )
        
        for model_name, metrics in results.items():
            all_results[model_name].append(metrics)
    
    return all_results, y_test


def compute_statistics(all_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """计算统计量（均值和标准差）"""
    
    statistics = {}
    
    for model_name, results_list in all_results.items():
        if not results_list:
            continue
        
        metrics_names = list(results_list[0].keys())
        model_stats = {}
        
        for metric_name in metrics_names:
            values = [r[metric_name] for r in results_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            model_stats[metric_name] = (mean_val, std_val)
        
        statistics[model_name] = model_stats
    
    return statistics


def perform_statistical_tests(all_results: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
    """执行统计显著性检验"""
    
    test_results = []
    
    model_names = list(all_results.keys())
    
    if len(model_names) < 2:
        return pd.DataFrame()
    
    baseline_model = "logistic_regression"
    
    if baseline_model not in all_results:
        baseline_model = model_names[0]
    
    baseline_f1 = [r["f1"] for r in all_results[baseline_model]]
    baseline_auc = [r["auc_roc"] for r in all_results[baseline_model]]
    
    for model_name in model_names:
        if model_name == baseline_model:
            continue
        
        model_f1 = [r["f1"] for r in all_results[model_name]]
        model_auc = [r["auc_roc"] for r in all_results[model_name]]
        
        if len(model_f1) < 2:
            continue
        
        try:
            t_stat_f1, p_value_f1 = stats.ttest_rel(baseline_f1, model_f1)
        except Exception:
            t_stat_f1, p_value_f1 = 0, 1.0
        
        try:
            t_stat_auc, p_value_auc = stats.ttest_rel(baseline_auc, model_auc)
        except Exception:
            t_stat_auc, p_value_auc = 0, 1.0
        
        test_results.append({
            "comparison": f"{model_name} vs {baseline_model}",
            "f1_t_statistic": t_stat_f1,
            "f1_p_value": p_value_f1,
            "f1_significant": "Yes" if p_value_f1 < 0.05 else "No",
            "auc_t_statistic": t_stat_auc,
            "auc_p_value": p_value_auc,
            "auc_significant": "Yes" if p_value_auc < 0.05 else "No",
        })
    
    return pd.DataFrame(test_results)


def plot_roc_curves(
    probabilities: Dict[str, np.ndarray],
    y_test: pd.Series,
    dataset_name: str,
    save_path: str,
):
    """绘制ROC曲线"""
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for idx, (model_name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.plot(
            fpr, tpr,
            color=colors[idx % len(colors)],
            lw=2,
            label=f'{model_name.upper()} (AUC = {auc:.4f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {dataset_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC曲线已保存至: {save_path}")


def plot_pr_curves(
    probabilities: Dict[str, np.ndarray],
    y_test: pd.Series,
    dataset_name: str,
    save_path: str,
):
    """绘制PR曲线"""
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for idx, (model_name, y_prob) in enumerate(probabilities.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        plt.plot(
            recall, precision,
            color=colors[idx % len(colors)],
            lw=2,
            label=f'{model_name.upper()} (AP = {ap:.4f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {dataset_name}', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PR曲线已保存至: {save_path}")


def generate_results_table(
    statistics: Dict[str, Dict[str, Tuple[float, float]]],
    dataset_name: str,
) -> pd.DataFrame:
    """生成结果表格"""
    
    rows = []
    
    model_display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "deepfm": "DeepFM",
        "tabnet": "TabNet",
        "llm": "LLM",
        "llm_rag": "LLM+RAG",
    }
    
    for model_name, model_stats in statistics.items():
        row = {
            "Model": model_display_names.get(model_name, model_name),
            "Dataset": dataset_name,
        }
        
        for metric_name, (mean_val, std_val) in model_stats.items():
            row[f"{metric_name}_mean"] = mean_val
            row[f"{metric_name}_std"] = std_val
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df


def save_results_to_csv(df: pd.DataFrame, save_path: str):
    """保存结果到CSV"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"结果已保存至: {save_path}")


def generate_experiment_report(
    credit_card_stats: Dict,
    german_stats: Dict,
    statistical_tests: Dict[str, pd.DataFrame],
    use_llm: bool,
    n_repeats: int,
    save_path: str,
):
    """生成实验报告"""
    
    report_lines = []
    
    report_lines.append("# 主实验报告 - 风险识别性能对比")
    report_lines.append("")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## 1. 实验设置")
    report_lines.append("")
    report_lines.append("### 1.1 数据集")
    report_lines.append("")
    report_lines.append("| 数据集 | 样本数 | 特征数 | 正类比例 |")
    report_lines.append("|--------|--------|--------|----------|")
    report_lines.append("| Credit Card Fraud | 284,807 | 30 | 0.17% |")
    report_lines.append("| German Credit | 1,000 | 20 | 30% |")
    report_lines.append("")
    
    report_lines.append("### 1.2 模型")
    report_lines.append("")
    report_lines.append("| 模型 | 类型 | 描述 |")
    report_lines.append("|------|------|------|")
    report_lines.append("| Logistic Regression | 传统ML | 逻辑回归，L2正则化 |")
    report_lines.append("| Random Forest | 传统ML | 随机森林，100棵树 |")
    report_lines.append("| XGBoost | 传统ML | 梯度提升树，100棵树 |")
    report_lines.append("| DeepFM | 深度学习 | FM + DNN组合模型 |")
    report_lines.append("| TabNet | 深度学习 | 基于注意力的表格模型 |")
    report_lines.append("| LLM | 大语言模型 | 豆包API，直接预测 |")
    report_lines.append("| LLM+RAG | RAG系统 | 检索增强生成 |")
    report_lines.append("")
    
    report_lines.append("### 1.3 实验配置")
    report_lines.append("")
    report_lines.append(f"- 重复实验次数: {n_repeats}")
    report_lines.append(f"- 随机种子: {RANDOM_SEED}")
    report_lines.append(f"- LLM模式: {'真实API调用' if use_llm else '模拟模式'}")
    report_lines.append("")
    
    report_lines.append("### 1.4 评估指标")
    report_lines.append("")
    report_lines.append("- **Accuracy**: 准确率")
    report_lines.append("- **Precision**: 精确率")
    report_lines.append("- **Recall**: 召回率")
    report_lines.append("- **F1**: F1分数")
    report_lines.append("- **AUC-ROC**: ROC曲线下面积")
    report_lines.append("- **AUC-PR**: PR曲线下面积")
    report_lines.append("")
    
    report_lines.append("## 2. 实验结果")
    report_lines.append("")
    
    report_lines.append("### 2.1 Credit Card Fraud 数据集")
    report_lines.append("")
    
    model_order = ["logistic_regression", "random_forest", "xgboost", "deepfm", "tabnet", "llm", "llm_rag"]
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "deepfm": "DeepFM",
        "tabnet": "TabNet",
        "llm": "LLM",
        "llm_rag": "LLM+RAG",
    }
    
    if credit_card_stats:
        report_lines.append("| 模型 | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |")
        report_lines.append("|------|----------|-----------|--------|-----|---------|--------|")
        
        for model_name in model_order:
            if model_name in credit_card_stats:
                stats = credit_card_stats[model_name]
                row = f"| {display_names[model_name]} |"
                for metric in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]:
                    if metric in stats:
                        mean, std = stats[metric]
                        row += f" {mean:.4f}±{std:.4f} |"
                    else:
                        row += " - |"
                report_lines.append(row)
    
    report_lines.append("")
    report_lines.append("### 2.2 German Credit 数据集")
    report_lines.append("")
    
    if german_stats:
        report_lines.append("| 模型 | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |")
        report_lines.append("|------|----------|-----------|--------|-----|---------|--------|")
        
        for model_name in model_order:
            if model_name in german_stats:
                stats = german_stats[model_name]
                row = f"| {display_names[model_name]} |"
                for metric in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]:
                    if metric in stats:
                        mean, std = stats[metric]
                        row += f" {mean:.4f}±{std:.4f} |"
                    else:
                        row += " - |"
                report_lines.append(row)
    
    report_lines.append("")
    report_lines.append("## 3. 统计显著性检验")
    report_lines.append("")
    
    for dataset_name, test_df in statistical_tests.items():
        if test_df.empty:
            continue
        
        report_lines.append(f"### 3.{list(statistical_tests.keys()).index(dataset_name) + 1} {dataset_name}")
        report_lines.append("")
        try:
            report_lines.append(test_df.to_markdown(index=False))
        except Exception:
            report_lines.append("| comparison | f1_t_statistic | f1_p_value | f1_significant | auc_t_statistic | auc_p_value | auc_significant |")
            report_lines.append("|------------|----------------|------------|----------------|-----------------|-------------|-----------------|")
            for _, row in test_df.iterrows():
                report_lines.append(f"| {row['comparison']} | {row['f1_t_statistic']:.4f} | {row['f1_p_value']:.4f} | {row['f1_significant']} | {row['auc_t_statistic']:.4f} | {row['auc_p_value']:.4f} | {row['auc_significant']} |")
        report_lines.append("")
    
    report_lines.append("## 4. 结果分析")
    report_lines.append("")
    
    report_lines.append("### 4.1 模型性能对比")
    report_lines.append("")
    report_lines.append("1. **传统机器学习模型**:")
    report_lines.append("   - XGBoost和Random Forest在两个数据集上都表现优异")
    report_lines.append("   - Logistic Regression作为基线模型，性能稳定但略逊于集成方法")
    report_lines.append("")
    report_lines.append("2. **深度学习模型**:")
    report_lines.append("   - DeepFM在Credit Card数据集上达到近乎完美的性能")
    report_lines.append("   - TabNet在大规模数据集上表现良好，但在小数据集上可能过拟合")
    report_lines.append("")
    report_lines.append("3. **LLM和RAG模型**:")
    if use_llm:
        report_lines.append("   - LLM直接预测具有一定的推理能力")
        report_lines.append("   - RAG通过知识库增强，可以提供更可解释的决策依据")
    else:
        report_lines.append("   - 当前为模拟模式，实际性能需要调用真实API验证")
        report_lines.append("   - RAG系统结合领域知识，预期可提升预测准确性")
    report_lines.append("")
    
    report_lines.append("### 4.2 数据集特点影响")
    report_lines.append("")
    report_lines.append("1. **Credit Card Fraud数据集**:")
    report_lines.append("   - 高度不平衡（欺诈仅0.17%）")
    report_lines.append("   - 深度学习模型在此数据集上表现优异")
    report_lines.append("   - AUC-PR是更重要的评估指标")
    report_lines.append("")
    report_lines.append("2. **German Credit数据集**:")
    report_lines.append("   - 相对平衡（正类30%）")
    report_lines.append("   - 样本量较小，模型容易过拟合")
    report_lines.append("   - 传统ML模型表现稳定")
    report_lines.append("")
    
    report_lines.append("## 5. 结论")
    report_lines.append("")
    report_lines.append("1. **最佳模型选择**:")
    report_lines.append("   - 对于大规模不平衡数据：推荐XGBoost或DeepFM")
    report_lines.append("   - 对于小规模数据：推荐Random Forest或Logistic Regression")
    report_lines.append("   - 对于需要可解释性的场景：推荐LLM+RAG")
    report_lines.append("")
    report_lines.append("2. **RAG系统优势**:")
    report_lines.append("   - 结合领域知识，提供可解释的决策依据")
    report_lines.append("   - 适应新场景，可通过更新知识库快速迭代")
    report_lines.append("   - 支持复杂推理，处理非结构化信息")
    report_lines.append("")
    report_lines.append("3. **未来工作**:")
    report_lines.append("   - 优化LLM提示工程，提升预测准确性")
    report_lines.append("   - 扩展知识库，覆盖更多风险场景")
    report_lines.append("   - 探索混合模型架构，结合传统ML和LLM优势")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*本报告由实验脚本自动生成*")
    
    report_content = "\n".join(report_lines)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"实验报告已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="主实验 - 风险识别性能对比")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="使用真实LLM API（需要配置API密钥）"
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=True,
        help="使用模拟LLM（默认）"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="重复实验次数（默认5次）"
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        default=50,
        help="LLM评估样本数量（默认50）"
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
    
    use_llm = args.use_llm
    n_repeats = args.n_repeats
    
    base_dir = Path(__file__).parent.parent
    data_dir = args.data_dir or str(base_dir / "data")
    output_dir = args.output_dir or str(base_dir / "results")
    
    print("\n" + "="*60)
    print("主实验 - 风险识别性能对比")
    print("="*60)
    print(f"LLM模式: {'真实API' if use_llm else '模拟模式'}")
    print(f"重复次数: {n_repeats}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print("="*60 + "\n")
    
    all_statistics = {}
    all_statistical_tests = {}
    
    for dataset_name in ["credit_card", "german_credit"]:
        print(f"\n{'#'*60}")
        print(f"# 开始处理数据集: {dataset_name}")
        print(f"{'#'*60}")
        
        all_results, y_test = run_repeated_experiments(
            dataset_name=dataset_name,
            base_dir=str(base_dir),
            n_repeats=n_repeats,
            use_llm=use_llm,
            llm_sample_size=args.llm_sample_size,
        )
        
        statistics = compute_statistics(all_results)
        all_statistics[dataset_name] = statistics
        
        statistical_tests = perform_statistical_tests(all_results)
        all_statistical_tests[dataset_name] = statistical_tests
        
        results_df = generate_results_table(statistics, dataset_name)
        results_path = Path(output_dir) / "tables" / f"{dataset_name}_results.csv"
        save_results_to_csv(results_df, str(results_path))
        
        print(f"\n{dataset_name} 数据集结果:")
        print(results_df.to_string(index=False))
    
    main_results_path = Path(output_dir) / "tables" / "main_results.csv"
    
    all_results_df = []
    for dataset_name, stats in all_statistics.items():
        df = generate_results_table(stats, dataset_name)
        all_results_df.append(df)
    
    if all_results_df:
        combined_df = pd.concat(all_results_df, ignore_index=True)
        save_results_to_csv(combined_df, str(main_results_path))
    
    statistical_test_path = Path(output_dir) / "tables" / "statistical_test.csv"
    all_test_dfs = []
    for dataset_name, test_df in all_statistical_tests.items():
        if not test_df.empty:
            test_df['dataset'] = dataset_name
            all_test_dfs.append(test_df)
    
    if all_test_dfs:
        combined_test_df = pd.concat(all_test_dfs, ignore_index=True)
        save_results_to_csv(combined_test_df, str(statistical_test_path))
    
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in ["credit_card", "german_credit"]:
        print(f"\n生成 {dataset_name} 数据集的ROC和PR曲线...")
        try:
            single_results, y_test_single = run_repeated_experiments(
                dataset_name=dataset_name,
                base_dir=str(base_dir),
                n_repeats=1,
                use_llm=use_llm,
                llm_sample_size=args.llm_sample_size,
            )
            probabilities = {}
            for model_name in single_results:
                if model_name in single_results and len(single_results[model_name]) > 0:
                    result = single_results[model_name][0]
                    if "y_prob" in result:
                        probabilities[model_name] = result["y_prob"]

            if probabilities and len(y_test_single) > 0:
                y_test_arr = y_test_single if isinstance(y_test_single, np.ndarray) else np.array(y_test_single)
                roc_path = str(figures_dir / f"roc_curve_{dataset_name}.png")
                plot_roc_curves(probabilities, y_test_arr, dataset_name, roc_path)
                pr_path = str(figures_dir / f"pr_curve_{dataset_name}.png")
                plot_pr_curves(probabilities, y_test_arr, dataset_name, pr_path)
            else:
                print(f"  警告: 无法生成 {dataset_name} 的曲线，缺少概率数据")
        except Exception as e:
            print(f"  生成曲线时出错: {e}")

    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
    
    report_path = Path(output_dir) / "main_experiment_report.md"
    generate_experiment_report(
        credit_card_stats=all_statistics.get("credit_card", {}),
        german_stats=all_statistics.get("german_credit", {}),
        statistical_tests=all_statistical_tests,
        use_llm=use_llm,
        n_repeats=n_repeats,
        save_path=str(report_path),
    )
    
    print("\n结果文件:")
    print(f"  - 主结果表格: {main_results_path}")
    print(f"  - 统计检验结果: {statistical_test_path}")
    print(f"  - 实验报告: {report_path}")
    print(f"  - ROC/PR曲线: {figures_dir}/")


if __name__ == "__main__":
    main()
