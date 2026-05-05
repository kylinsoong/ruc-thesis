import sys
import os
import json
import time
import re
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from api_client import DoubaoAPIClient


RISK_ANALYSIS_PROMPT = """你是一位资深的金融风控专家。请根据以下客户特征数据进行风险评估。

## 客户特征
{feature_str}

## 评估要求
1. 分析各特征的风险信号
2. 给出风险评分（0-1之间，1表示高风险）
3. 给出审批建议（批准/拒绝）

## 输出格式
请严格按照以下JSON格式输出：
{{
    "risk_score": 0.0到1.0之间的风险评分,
    "recommendation": "批准"或"拒绝"
}}
"""


class LLMOnlyModel:
    def __init__(
        self,
        api_client: DoubaoAPIClient = None,
        temperature: float = 0.3,
        max_retries: int = 3,
        mock_mode: bool = False,
        random_state: int = 42,
        n_workers: int = 10
    ):
        self.api_client = api_client or DoubaoAPIClient()
        self.temperature = temperature
        self.max_retries = max_retries
        self.mock_mode = mock_mode
        self.random_state = random_state
        self.n_workers = n_workers
        np.random.seed(random_state)
        self.is_fitted = True

    def _build_prompt(self, features: Dict[str, any], feature_names: List[str]) -> str:
        feature_str = "\n".join([
            f"- {name}: {features[name]}"
            for name in feature_names
        ])
        return RISK_ANALYSIS_PROMPT.format(feature_str=feature_str)

    def _parse_response(self, response_text: str) -> tuple:
        try:
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                risk_score = float(result.get("risk_score", 0.5))
                recommendation = result.get("recommendation", "批准")

                label = 0 if recommendation == "批准" else 1
                return risk_score, label, True
            else:
                return 0.5, 0, False
        except (json.JSONDecodeError, ValueError, KeyError):
            return 0.5, 0, False

    def _mock_predict_single(self, features: Dict, feature_names: List[str]) -> tuple:
        base_prob = 0.3 + np.random.random() * 0.4
        risk_score = min(1.0, max(0.0, base_prob + np.random.randn() * 0.1))
        label = 1 if risk_score > 0.5 else 0
        return risk_score, label

    def predict_single(self, features: Union[np.ndarray, Dict, List]) -> tuple:
        if isinstance(features, np.ndarray):
            features = features.tolist()

        if isinstance(features, list):
            features = {f"feature_{i}": v for i, v in enumerate(features)}

        if self.mock_mode:
            feature_names = list(features.keys())
            return self._mock_predict_single(features, feature_names)

        feature_names = list(features.keys())
        prompt = self._build_prompt(features, feature_names)

        for attempt in range(self.max_retries):
            try:
                response = self.api_client.call_llm(prompt, temperature=self.temperature)
                content = response.get("content", "")

                if content:
                    risk_score, label, success = self._parse_response(content)
                    if success:
                        return risk_score, label

                time.sleep(0.5)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return 0.5, 0
                time.sleep(1)

        return 0.5, 0

    def _predict_batch_parallel(self, X: Union[pd.DataFrame, np.ndarray], return_proba: bool = False) -> np.ndarray:
        if isinstance(X, np.ndarray):
            samples = [{f"feature_{i}": v for i, v in enumerate(row)} for row in X]
        else:
            samples = [row.to_dict() for _, row in X.iterrows()]

        results = [None] * len(samples)
        completed = 0
        total = len(samples)
        start_time = time.time()

        def worker(idx, sample):
            if return_proba:
                return idx, self.predict_single(sample)[0]
            else:
                return idx, self.predict_single(sample)[1]

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(worker, i, s): i for i, s in enumerate(samples)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                print(f"\r  进度: {completed}/{total} ({completed*100//total}%) | 已用: {elapsed:.0f}s | 剩余: ~{eta:.0f}s", end="", flush=True)

        print()
        return np.array(results)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.mock_mode:
            if isinstance(X, np.ndarray):
                n_samples = X.shape[0]
                predictions = []
                for i in range(n_samples):
                    _, label = self.predict_single(X[i].tolist())
                    predictions.append(label)
                    print(f"\r  进度: {i+1}/{n_samples} ({(i+1)*100//n_samples}%)", end="", flush=True)
                print()
                return np.array(predictions)

            from pandas import DataFrame
            if isinstance(X, DataFrame):
                n_samples = len(X)
                predictions = []
                for idx, row in X.iterrows():
                    _, label = self.predict_single(row.to_dict())
                    predictions.append(label)
                    print(f"\r  进度: {len(predictions)}/{n_samples} ({len(predictions)*100//n_samples}%)", end="", flush=True)
                print()
                return np.array(predictions)

            return np.array([])

        return self._predict_batch_parallel(X, return_proba=False)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.mock_mode:
            if isinstance(X, np.ndarray):
                n_samples = X.shape[0]
                probabilities = []
                for i in range(n_samples):
                    risk_score, _ = self.predict_single(X[i].tolist())
                    probabilities.append(risk_score)
                    print(f"\r  进度: {i+1}/{n_samples} ({(i+1)*100//n_samples}%)", end="", flush=True)
                print()
                return np.array(probabilities)

            from pandas import DataFrame
            if isinstance(X, DataFrame):
                n_samples = len(X)
                probabilities = []
                for idx, row in X.iterrows():
                    risk_score, _ = self.predict_single(row.to_dict())
                    probabilities.append(risk_score)
                    print(f"\r  进度: {len(probabilities)}/{n_samples} ({len(probabilities)*100//n_samples}%)", end="", flush=True)
                print()
                return np.array(probabilities)

            return np.array([])

        return self._predict_batch_parallel(X, return_proba=True)

    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame], y_test: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        n_samples = len(X_test)
        print(f"  正在预测标签 (共{n_samples}条样本)...")
        y_pred = self.predict(X_test)
        print(f"  正在预测概率 (共{n_samples}条样本)...")
        y_prob = self.predict_proba(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        try:
            metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics["auc_roc"] = 0.0

        try:
            metrics["auc_pr"] = float(average_precision_score(y_test, y_prob))
        except ValueError:
            metrics["auc_pr"] = 0.0

        return metrics
