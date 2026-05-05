"""
基线模型模块

包含以下模型：
- 传统机器学习模型：逻辑回归、随机森林、XGBoost
- 深度学习模型：DeepFM、TabNet
"""

import warnings

BaseModel = None
LogisticRegressionModel = None
RandomForestModel = None
XGBoostModel = None
create_model = None
get_available_models = None
DeepFMModel = None
TabNetModel = None
create_deep_learning_model = None
DeepFM = None
FMComponent = None
DeepComponent = None

try:
    from models.baseline.traditional_ml import (
        BaseModel,
        LogisticRegressionModel,
        RandomForestModel,
        XGBoostModel,
        create_model,
        get_available_models,
    )
except ImportError as e:
    warnings.warn(f"Could not import traditional ML models: {e}")

try:
    from models.baseline.deep_learning import (
        DeepFMModel,
        TabNetModel,
        create_deep_learning_model,
        DeepFM,
        FMComponent,
        DeepComponent,
    )
except ImportError as e:
    warnings.warn(f"Could not import deep learning models: {e}")

__all__ = [
    "BaseModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "create_model",
    "get_available_models",
    "DeepFMModel",
    "TabNetModel",
    "create_deep_learning_model",
    "DeepFM",
    "FMComponent",
    "DeepComponent",
]
