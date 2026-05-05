import os
from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIG = {
    "llm": {
        "model_name": "doubao-seed-2-0-pro-260215",
        "api_endpoint": "https://ark.cn-beijing.volces.com/api/v3",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "embedding": {
        "model_name": "doubao-embedding-vision-251215",
        "api_endpoint": "https://ark.cn-beijing.volces.com/api/v3",
        "dimension": 2048,
    }
}

API_CONFIG = {
    "access_key": os.getenv("VOLCENGINE_ACCESS_KEY", ""),
    "secret_key": os.getenv("VOLCENGINE_SECRET_KEY", ""),
    "endpoint_id": os.getenv("VOLCENGINE_ENDPOINT_ID", ""),
}

DATA_CONFIG = {
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "knowledge_base_dir": "data/knowledge_base",
}

EXPERIMENT_CONFIG = {
    "results_dir": "results",
    "tables_dir": "results/tables",
    "figures_dir": "results/figures",
    "random_seed": 42,
}

RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,
    "similarity_threshold": 0.7,
}
