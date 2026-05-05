import pandas as pd
from pathlib import Path
from typing import Tuple


def load_credit_card_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "processed"
    else:
        data_dir = Path(data_dir) / "data" / "processed"

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def load_german_credit_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "processed"
    else:
        data_dir = Path(data_dir) / "data" / "processed"

    X_train = pd.read_csv(data_dir / "german_X_train.csv")
    X_test = pd.read_csv(data_dir / "german_X_test.csv")
    y_train = pd.read_csv(data_dir / "german_y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "german_y_test.csv").squeeze()

    if y_train.min() == 1 and y_train.max() == 2:
        y_train = y_train - 1
        y_test = y_test - 1

    return X_train, X_test, y_train, y_test


def load_credit_card_llm_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "processed"
    else:
        data_dir = Path(data_dir) / "data" / "processed"

    X_train = pd.read_csv(data_dir / "credit_card_llm_X_train.csv")
    X_test = pd.read_csv(data_dir / "credit_card_llm_X_test.csv")
    y_train = pd.read_csv(data_dir / "credit_card_llm_y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "credit_card_llm_y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def load_data(dataset_name: str, data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if dataset_name.lower() in ["credit_card", "credit_card_fraud"]:
        return load_credit_card_data(data_dir)
    elif dataset_name.lower() in ["credit_card_llm"]:
        return load_credit_card_llm_data(data_dir)
    elif dataset_name.lower() in ["german_credit", "german"]:
        return load_german_credit_data(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: credit_card, credit_card_llm, german_credit")
