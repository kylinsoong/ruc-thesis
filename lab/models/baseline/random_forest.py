import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


class RandomForestModel:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = SkRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names = None

    def train(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series]) -> None:
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame], y_test: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        try:
            metrics["auc_roc"] = roc_auc_score(y_test, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0

        try:
            metrics["auc_pr"] = average_precision_score(y_test, y_prob)
        except ValueError:
            metrics["auc_pr"] = 0.0

        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}

        importance_dict = {}
        if self.feature_names is not None:
            for name, imp in zip(self.feature_names, self.model.feature_importances_):
                importance_dict[name] = float(imp)
        else:
            n_features = len(self.model.feature_importances_)
            for i in range(n_features):
                importance_dict[f"feature_{i}"] = float(self.model.feature_importances_[i])

        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
