import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from utils.model_utils import save_model, calculate_metrics


class BaselineModel:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "xgboost":
            return XGBClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath: str):
        save_model(self.model, filepath)


class RuleBasedModel:
    def __init__(self, rules: dict = None):
        self.rules = rules or {}

    def apply_rules(self, data):
        pass
        return []

    def predict(self, data):
        return self.apply_rules(data)
