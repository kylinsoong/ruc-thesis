import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


class LogisticRegressionModel:
    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42
    ):
        self.model = SkLogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state
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

    def get_coefficients(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}

        coef_dict = {}
        if self.feature_names is not None:
            for name, coef in zip(self.feature_names, self.model.coef_[0]):
                coef_dict[name] = float(coef)
        else:
            n_features = len(self.model.coef_[0])
            for i in range(n_features):
                coef_dict[f"feature_{i}"] = float(self.model.coef_[0][i])

        return coef_dict
