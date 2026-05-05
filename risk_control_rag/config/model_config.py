"""
模型超参数配置模块

包含以下模型的超参数配置：
- Logistic Regression
- Random Forest
- XGBoost
"""

from typing import Dict, Any


LOGISTIC_REGRESSION_CONFIG = {
    "default": {
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    },
    "credit_card": {
        "penalty": "l2",
        "C": 0.1,
        "max_iter": 2000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    },
    "german_credit": {
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    },
}


RANDOM_FOREST_CONFIG = {
    "default": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "credit_card": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "german_credit": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
}


XGBOOST_CONFIG = {
    "default": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": None,
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    },
    "credit_card": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": None,
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "auc",
    },
    "german_credit": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": None,
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    },
}


MODEL_CONFIGS = {
    "logistic_regression": LOGISTIC_REGRESSION_CONFIG,
    "random_forest": RANDOM_FOREST_CONFIG,
    "xgboost": XGBOOST_CONFIG,
}


def get_model_config(model_type: str, dataset_name: str = "default") -> Dict[str, Any]:
    """
    获取指定模型和数据集的超参数配置
    
    Args:
        model_type: 模型类型，可选 'logistic_regression', 'random_forest', 'xgboost'
        dataset_name: 数据集名称，可选 'default', 'credit_card', 'german_credit'
        
    Returns:
        Dict[str, Any]: 超参数配置字典
        
    Raises:
        ValueError: 不支持的模型类型
    """
    model_type = model_type.lower()
    
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"不支持的模型类型: {model_type}\n"
            f"支持的模型类型: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = MODEL_CONFIGS[model_type]
    
    if dataset_name in config:
        return config[dataset_name].copy()
    else:
        print(f"警告: 数据集 '{dataset_name}' 配置不存在，使用默认配置")
        return config["default"].copy()


def get_all_model_configs() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """返回所有模型配置"""
    return MODEL_CONFIGS


def list_available_configs() -> Dict[str, list]:
    """列出所有可用的配置"""
    return {
        model_type: list(configs.keys())
        for model_type, configs in MODEL_CONFIGS.items()
    }
