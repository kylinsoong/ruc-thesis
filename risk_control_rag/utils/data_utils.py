import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None,
) -> tuple:
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)


def preprocess_features(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    df_processed = df.copy()
    
    scaler = StandardScaler()
    if numerical_cols:
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    return df_processed


def create_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def format_knowledge_entry(
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "title": title,
        "content": content,
        "metadata": metadata or {},
    }
    return entry
