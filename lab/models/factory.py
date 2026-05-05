from typing import Dict, Any


def create_model(model_type: str, **kwargs) -> Any:
    model_type = model_type.lower()

    if model_type in ["logistic_regression", "lr", "logistic"]:
        from .baseline.logistic_regression import LogisticRegressionModel
        return LogisticRegressionModel(**kwargs)

    elif model_type in ["random_forest", "rf"]:
        from .baseline.random_forest import RandomForestModel
        return RandomForestModel(**kwargs)

    elif model_type in ["xgboost", "xgb"]:
        from .baseline.xgboost_model import XGBoostModel
        return XGBoostModel(**kwargs)

    elif model_type in ["deepfm", "deep_fm"]:
        from .baseline.deepfm import TabularDeepFM
        return TabularDeepFM(**kwargs)

    elif model_type in ["tabnet"]:
        from .baseline.tabnet import SimpleTabNet
        return SimpleTabNet(**kwargs)

    elif model_type in ["llm", "llm_only", "large_language_model"]:
        from .baseline.llm_only import LLMOnlyModel
        return LLMOnlyModel(**kwargs)

    elif model_type in ["rag", "llm_rag"]:
        from .rag_model import RAGModel
        return RAGModel(**kwargs)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}\n"
            f"Supported models: logistic_regression, random_forest, xgboost, deepfm, tabnet, llm"
        )


def get_available_models() -> list:
    return ["logistic_regression", "random_forest", "xgboost", "deepfm", "tabnet", "llm", "rag"]


def get_model_display_name(model_type: str) -> str:
    display_names = {
        "logistic_regression": "Logistic Regression",
        "lr": "Logistic Regression",
        "random_forest": "Random Forest",
        "rf": "Random Forest",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
        "deepfm": "DeepFM",
        "deep_fm": "DeepFM",
        "tabnet": "TabNet",
        "llm": "LLM-only",
        "llm_only": "LLM-only",
        "large_language_model": "LLM-only",
        "rag": "RAG",
        "llm_rag": "LLM+RAG"
    }
    return display_names.get(model_type.lower(), model_type)
