"""
配置模块

包含以下配置：
- config: 基础配置（模型、API、数据、实验）
- model_config: 模型超参数配置
- prompts: Prompt模板配置
"""

from config.config import (
    MODEL_CONFIG,
    API_CONFIG,
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    RAG_CONFIG,
)
from config.model_config import (
    LOGISTIC_REGRESSION_CONFIG,
    RANDOM_FOREST_CONFIG,
    XGBOOST_CONFIG,
    MODEL_CONFIGS,
    get_model_config,
    get_all_model_configs,
    list_available_configs,
)
from config.prompts import (
    RISK_ANALYSIS_SYSTEM_PROMPT,
    RISK_ANALYSIS_PROMPT_TEMPLATE,
    get_risk_analysis_prompt,
    get_batch_risk_analysis_prompt,
    format_customer_info,
    parse_risk_level,
    extract_risk_score,
    extract_approval_suggestion,
)

__all__ = [
    "MODEL_CONFIG",
    "API_CONFIG",
    "DATA_CONFIG",
    "EXPERIMENT_CONFIG",
    "RAG_CONFIG",
    "LOGISTIC_REGRESSION_CONFIG",
    "RANDOM_FOREST_CONFIG",
    "XGBOOST_CONFIG",
    "MODEL_CONFIGS",
    "get_model_config",
    "get_all_model_configs",
    "list_available_configs",
    "RISK_ANALYSIS_SYSTEM_PROMPT",
    "RISK_ANALYSIS_PROMPT_TEMPLATE",
    "get_risk_analysis_prompt",
    "get_batch_risk_analysis_prompt",
    "format_customer_info",
    "parse_risk_level",
    "extract_risk_score",
    "extract_approval_suggestion",
]
