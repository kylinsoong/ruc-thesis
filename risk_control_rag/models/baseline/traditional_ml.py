"""
传统机器学习模型模块

包含以下模型：
- LogisticRegressionModel: 逻辑回归
- RandomForestModel: 随机森林
- XGBoostModel: XGBoost梯度提升树

每个模型都提供统一的接口：
- train(): 训练方法
- predict(): 预测方法
- predict_proba(): 概率预测方法
- get_feature_importance(): 特征重要性（适用于RF和XGBoost）
- save_model(): 模型保存
- load_model(): 模型加载
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import joblib
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class BaseModel:
    """模型基类，定义统一接口"""
    
    def __init__(self, model_name: str, random_state: int = 42, **kwargs):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
    
    def train(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series]) -> None:
        raise NotImplementedError("子类必须实现train方法")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        raise NotImplementedError("子类必须实现predict方法")
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        raise NotImplementedError("子类必须实现predict_proba方法")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None
    
    def save_model(self, path: str) -> None:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path: str) -> None:
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.model_name = model_data["model_name"]
        self.feature_names = model_data.get("feature_names")
        self.is_fitted = model_data["is_fitted"]
        print(f"模型已从 {path} 加载")
    
    def _extract_feature_names(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[List[str]]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return None


class LogisticRegressionModel(BaseModel):
    """逻辑回归模型"""
    
    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        **kwargs
    ):
        super().__init__("LogisticRegression", random_state, **kwargs)
        
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.class_weight = class_weight
        
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
    
    def train(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series]) -> None:
        self.feature_names = self._extract_feature_names(X_train)
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"逻辑回归模型训练完成")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted:
            return None
        
        if self.feature_names is None:
            n_features = len(self.model.coef_[0])
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        importance = np.abs(self.model.coef_[0])
        
        importance_dict = {}
        for name, imp in zip(self.feature_names, importance):
            importance_dict[name] = float(imp)
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_coefficients(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        
        if self.feature_names is None:
            n_features = len(self.model.coef_[0])
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        coef_dict = {}
        for name, coef in zip(self.feature_names, self.model.coef_[0]):
            coef_dict[name] = float(coef)
        
        return coef_dict


class RandomForestModel(BaseModel):
    """随机森林模型"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__("RandomForest", random_state, **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
    
    def train(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series]) -> None:
        self.feature_names = self._extract_feature_names(X_train)
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"随机森林模型训练完成 (n_estimators={self.n_estimators})")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted:
            return None
        
        if self.feature_names is None:
            n_features = len(self.model.feature_importances_)
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        importance_dict = {}
        for name, imp in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(imp)
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class XGBoostModel(BaseModel):
    """XGBoost梯度提升树模型"""
    
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
        eval_metric: str = "logloss",
        **kwargs
    ):
        super().__init__("XGBoost", random_state, **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        
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
            eval_metric=eval_metric,
            **kwargs
        )
    
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> None:
        self.feature_names = self._extract_feature_names(X_train)
        
        if isinstance(X_train, pd.DataFrame):
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
        print(f"XGBoost模型训练完成 (n_estimators={self.n_estimators})")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type: str = "gain") -> Optional[Dict[str, float]]:
        if not self.is_fitted:
            return None
        
        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(len(self.model.feature_importances_))]
        
        try:
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            
            importance_dict = {}
            for i, name in enumerate(self.feature_names):
                key = f"f{i}"
                importance_dict[name] = float(importance.get(key, 0.0))
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            importance_dict = {}
            for name, imp in zip(self.feature_names, self.model.feature_importances_):
                importance_dict[name] = float(imp)
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def create_model(model_type: str, **kwargs) -> BaseModel:
    """
    工厂函数：根据模型类型创建模型实例
    
    Args:
        model_type: 模型类型，可选 'logistic_regression', 'random_forest', 'xgboost'
        **kwargs: 模型参数
        
    Returns:
        BaseModel: 模型实例
        
    Raises:
        ValueError: 不支持的模型类型
    """
    model_type = model_type.lower()
    
    if model_type in ["logistic_regression", "lr", "logistic"]:
        return LogisticRegressionModel(**kwargs)
    elif model_type in ["random_forest", "rf"]:
        return RandomForestModel(**kwargs)
    elif model_type in ["xgboost", "xgb"]:
        return XGBoostModel(**kwargs)
    else:
        raise ValueError(
            f"不支持的模型类型: {model_type}\n"
            f"支持的模型类型: logistic_regression, random_forest, xgboost"
        )


def get_available_models() -> List[str]:
    """返回可用的模型类型列表"""
    return ["logistic_regression", "random_forest", "xgboost"]
