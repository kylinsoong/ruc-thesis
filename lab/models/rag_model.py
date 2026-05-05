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
from rag.hybrid_retriever import HybridRetriever
from rag.rag_system import FinancialRiskRAGSystem


class RAGModel:
    def __init__(
        self,
        api_client=None,
        hybrid_retriever=None,
        embedder=None,
        temperature=0.3,
        mock_mode=False,
        random_state=42,
        n_workers=10
    ):
        self.api_client = api_client or DoubaoAPIClient()
        self.hybrid_retriever = hybrid_retriever
        self.embedder = embedder
        self.temperature = temperature
        self.mock_mode = mock_mode
        self.random_state = random_state
        self.n_workers = n_workers
        np.random.seed(random_state)
        self.is_fitted = True
        self.feature_names_ = None
        self.rag_system_ = None

        if self.hybrid_retriever is not None and self.embedder is not None:
            self.rag_system_ = FinancialRiskRAGSystem(
                llm_client=self.api_client,
                hybrid_retriever=self.hybrid_retriever,
                embedder=self.embedder
            )

    def _build_features_dict(self, features: Union[np.ndarray, pd.DataFrame, Dict]) -> Dict:
        if isinstance(features, dict):
            return features
        if isinstance(features, np.ndarray):
            if self.feature_names_ is not None:
                return {name: value for name, value in zip(self.feature_names_, features)}
            return {f"feature_{i}": v for i, v in enumerate(features)}
        if isinstance(features, pd.DataFrame):
            return features.to_dict(orient='records')[0] if len(features) > 0 else {}
        return {}

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

        if self.rag_system_ is None:
            return 0.5, 0

        feature_names = list(features.keys())
        feature_values = list(features.values())

        try:
            result = self.rag_system_.predict(feature_values, feature_names)
            response_text = result.get("response", {}).get("content", "")
            if response_text:
                risk_score, label, success = self._parse_response(response_text)
                if success:
                    return risk_score, label
        except Exception as e:
            pass

        return 0.5, 0

    def _predict_batch_parallel(self, X: Union[pd.DataFrame, np.ndarray], return_proba: bool = False) -> np.ndarray:
        if isinstance(X, np.ndarray):
            samples = [{f"feature_{i}": v for i, v in enumerate(row)} for row in X]
        else:
            samples = [row.to_dict() for _, row in X.iterrows()]

        results = [None] * len(samples)

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

        return np.array(results)

    def train(self, X_train, y_train):
        self.is_fitted = True
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
        elif isinstance(X_train, np.ndarray) and X_train.ndim == 2:
            self.feature_names_ = [f"feature_{i}" for i in range(X_train.shape[1])]
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.mock_mode:
            if isinstance(X, np.ndarray):
                n_samples = X.shape[0]
                predictions = []
                for i in range(n_samples):
                    _, label = self.predict_single(X[i].tolist())
                    predictions.append(label)
                return np.array(predictions)

            from pandas import DataFrame
            if isinstance(X, DataFrame):
                predictions = []
                for idx, row in X.iterrows():
                    _, label = self.predict_single(row.to_dict())
                    predictions.append(label)
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
                return np.array(probabilities)

            from pandas import DataFrame
            if isinstance(X, DataFrame):
                probabilities = []
                for idx, row in X.iterrows():
                    risk_score, _ = self.predict_single(row.to_dict())
                    probabilities.append(risk_score)
                return np.array(probabilities)

            return np.array([])

        return self._predict_batch_parallel(X, return_proba=True)

    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame], y_test: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        y_pred = self.predict(X_test)
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
