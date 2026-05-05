import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


class XGBoostModel:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        use_label_encoder: bool = False,
        eval_metric: str = "logloss"
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            use_label_encoder=use_label_encoder,
            eval_metric=eval_metric
        )
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names = None

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = None
    ) -> None:
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
        if early_stopping_rounds is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X_train, y_train, **fit_params)
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
