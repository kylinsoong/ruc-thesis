"""
数据下载模块

提供数据集下载功能，包括：
- German Credit 数据集
- Credit Card Fraud 数据集
"""

import os
import urllib.request
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

GERMAN_CREDIT_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "class"
]

GERMAN_CREDIT_CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings_status", "employment",
    "personal_status", "other_parties", "property_magnitude", "other_payment_plans",
    "housing", "job", "own_telephone", "foreign_worker"
]

GERMAN_CREDIT_NUMERICAL = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents"
]


def download_german_credit(
    save_dir: str,
    filename: str = "german.data",
    force_download: bool = False,
) -> str:
    save_path = Path(save_dir) / filename
    
    if save_path.exists() and not force_download:
        print(f"数据集已存在: {save_path}")
        return str(save_path)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在下载German Credit数据集...")
    print(f"URL: {GERMAN_CREDIT_URL}")
    
    try:
        urllib.request.urlretrieve(GERMAN_CREDIT_URL, save_path)
        print(f"下载完成，保存至: {save_path}")
    except Exception as e:
        raise RuntimeError(f"下载失败: {e}")
    
    return str(save_path)


def load_german_credit(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=" ",
        header=None,
        names=GERMAN_CREDIT_COLUMNS,
    )
    return df


def get_german_credit(
    save_dir: str = "data/raw",
    force_download: bool = False,
) -> pd.DataFrame:
    filepath = download_german_credit(save_dir, force_download=force_download)
    df = load_german_credit(filepath)
    return df


def get_feature_info() -> dict:
    return {
        "categorical": GERMAN_CREDIT_CATEGORICAL,
        "numerical": GERMAN_CREDIT_NUMERICAL,
        "target": "class",
        "all_features": GERMAN_CREDIT_COLUMNS[:-1],
    }


def check_kaggle_api() -> bool:
    """
    检查Kaggle API是否可用。
    
    Returns:
        bool: Kaggle API是否正确配置
    """
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    return os.path.exists(kaggle_json)


def download_credit_card_fraud_data(
    save_dir: str,
    force_download: bool = False,
) -> Optional[str]:
    """
    从Kaggle下载Credit Card Fraud数据集。
    
    Args:
        save_dir: 数据保存目录
        force_download: 是否强制重新下载
        
    Returns:
        str: 下载文件的路径，如果下载失败返回None
        
    Raises:
        ImportError: 如果未安装kaggle库
        RuntimeError: 如果Kaggle API未正确配置
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    target_file = save_path / "creditcard.csv"
    
    if target_file.exists() and not force_download:
        print(f"数据文件已存在: {target_file}")
        return str(target_file)
    
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "未安装kaggle库。请运行: pip install kaggle\n"
            "并确保已配置Kaggle API凭证。"
        )
    
    if not check_kaggle_api():
        raise RuntimeError(
            "Kaggle API未正确配置。\n"
            "请按以下步骤操作：\n"
            "1. 登录 https://www.kaggle.com/\n"
            "2. 进入 Account 设置页面\n"
            "3. 点击 'Create New API Token' 下载 kaggle.json\n"
            "4. 将 kaggle.json 放置到 ~/.kaggle/ 目录\n"
            "5. 运行: chmod 600 ~/.kaggle/kaggle.json"
        )
    
    print("正在从Kaggle下载Credit Card Fraud数据集...")
    
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path=str(save_path),
            unzip=True,
        )
        print(f"数据集下载完成: {target_file}")
        return str(target_file)
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def get_manual_download_instructions() -> str:
    """
    获取手动下载数据集的说明。
    
    Returns:
        str: 手动下载说明
    """
    instructions = """
========================================
Credit Card Fraud 数据集手动下载说明
========================================

如果自动下载失败，请按以下步骤手动下载数据集：

1. 访问 Kaggle 数据集页面:
   https://www.kaggle.com/mlg-ulb/creditcardfraud

2. 点击 "Download" 按钮下载 creditcardfraud.zip

3. 解压下载的文件，获得 creditcard.csv

4. 将 creditcard.csv 放置到以下目录:
   risk_control_rag/data/raw/creditcard.csv

数据集说明：
- 该数据集包含2013年9月欧洲持卡人的信用卡交易
- 共284,807条交易记录，其中492条为欺诈交易
- 数据集高度不平衡，欺诈交易仅占0.172%
- 特征V1-V28是PCA转换后的主成分
- 'Time'列表示每笔交易与第一笔交易的时间差（秒）
- 'Amount'列表示交易金额
- 'Class'列是目标变量：0=正常，1=欺诈

注意事项：
- 下载需要Kaggle账号
- 请遵守数据集的使用许可协议
========================================
"""
    return instructions


def load_credit_card_data(filepath: str) -> pd.DataFrame:
    """
    加载Credit Card Fraud数据集。
    
    Args:
        filepath: 数据文件路径
        
    Returns:
        pd.DataFrame: 加载的数据框
        
    Raises:
        FileNotFoundError: 如果文件不存在
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"数据文件不存在: {filepath}\n"
            f"请先下载数据集。{get_manual_download_instructions()}"
        )
    
    print(f"正在加载数据: {filepath}")
    df = pd.read_csv(filepath)
    print(f"数据加载完成，共 {len(df)} 条记录")
    return df


def verify_data_integrity(df: pd.DataFrame) -> dict:
    """
    验证数据完整性。
    
    Args:
        df: 数据框
        
    Returns:
        dict: 数据验证结果
    """
    expected_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    
    results = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "has_all_columns": all(col in df.columns for col in expected_columns),
        "missing_columns": [col for col in expected_columns if col not in df.columns],
        "has_class_column": 'Class' in df.columns,
        "class_values": df['Class'].unique().tolist() if 'Class' in df.columns else [],
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
    }
    
    return results


if __name__ == "__main__":
    print(get_manual_download_instructions())
