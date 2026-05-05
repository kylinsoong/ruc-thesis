"""
数据预处理模块

提供Credit Card Fraud数据集的预处理功能，包括：
- 缺失值处理
- 特征标准化
- 类别不平衡处理（SMOTE过采样）
- 数据集划分
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查并报告缺失值情况。
    
    Args:
        df: 输入数据框
        
    Returns:
        pd.DataFrame: 缺失值统计报告
    """
    missing_report = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_ratio': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values,
    })
    missing_report = missing_report.sort_values('missing_count', ascending=False)
    return missing_report


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    处理缺失值。
    
    Args:
        df: 输入数据框
        strategy: 填充策略，可选 'mean', 'median', 'most_frequent', 'constant'
        columns: 需要处理的列，默认处理所有数值列
        
    Returns:
        Tuple[pd.DataFrame, Dict]: 处理后的数据框和处理信息
    """
    df_processed = df.copy()
    processing_info = {
        "strategy": strategy,
        "columns_processed": [],
        "values_filled": {},
    }
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if df_processed[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy=strategy)
            df_processed[col] = imputer.fit_transform(df_processed[[col]])
            processing_info["columns_processed"].append(col)
            if strategy == 'mean':
                processing_info["values_filled"][col] = float(imputer.statistics_[0])
    
    return df_processed, processing_info


def standardize_features(
    df: pd.DataFrame,
    columns: list,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    对指定列进行标准化处理。
    
    Args:
        df: 输入数据框
        columns: 需要标准化的列名列表
        scaler: 预拟合的StandardScaler，如果为None则创建新的
        fit: 是否拟合scaler
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: 标准化后的数据框和scaler
    """
    df_standardized = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        df_standardized[columns] = scaler.fit_transform(df[columns])
    else:
        df_standardized[columns] = scaler.transform(df[columns])
    
    return df_standardized, scaler


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    sampling_strategy: str = 'auto',
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    应用SMOTE过采样处理类别不平衡。
    
    Args:
        X: 特征数据框
        y: 目标变量
        random_state: 随机种子
        sampling_strategy: 采样策略
        
    Returns:
        Tuple: 过采样后的特征、目标变量和处理信息
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "未安装imbalanced-learn库。\n"
            "请运行: pip install imbalanced-learn"
        )
    
    original_distribution = y.value_counts().to_dict()
    
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    resampled_distribution = pd.Series(y_resampled).value_counts().to_dict()
    
    processing_info = {
        "original_distribution": original_distribution,
        "resampled_distribution": resampled_distribution,
        "original_samples": len(y),
        "resampled_samples": len(y_resampled),
        "samples_added": len(y_resampled) - len(y),
    }
    
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled), processing_info


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    划分训练集和测试集。
    
    Args:
        X: 特征数据框
        y: 目标变量
        test_size: 测试集比例
        random_state: 随机种子
        stratify: 是否分层抽样
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )
    
    return X_train, X_test, y_train, y_test


def preprocess_credit_card_data(
    df: pd.DataFrame,
    target_column: str = 'Class',
    standardize_cols: list = ['Amount', 'Time'],
    apply_smote_oversampling: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Credit Card Fraud数据集完整预处理流程。
    
    Args:
        df: 原始数据框
        target_column: 目标列名
        standardize_cols: 需要标准化的列
        apply_smote_oversampling: 是否应用SMOTE过采样
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        Dict: 包含处理后的数据和预处理信息的字典
    """
    preprocessing_info = {
        "original_shape": df.shape,
        "steps": [],
    }
    
    df_processed = df.copy()
    
    missing_report = check_missing_values(df_processed)
    preprocessing_info["missing_report"] = missing_report.to_dict()
    
    if df_processed.isnull().sum().sum() > 0:
        df_processed, missing_info = handle_missing_values(df_processed)
        preprocessing_info["steps"].append({
            "name": "missing_value_handling",
            "info": missing_info,
        })
    
    feature_columns = [col for col in df_processed.columns if col != target_column]
    X = df_processed[feature_columns]
    y = df_processed[target_column]
    
    scaler = StandardScaler()
    X[standardize_cols] = scaler.fit_transform(X[standardize_cols])
    preprocessing_info["steps"].append({
        "name": "standardization",
        "info": {
            "columns": standardize_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
    })
    
    smote_info = None
    if apply_smote_oversampling:
        X, y, smote_info = apply_smote(X, y, random_state=random_state)
        preprocessing_info["steps"].append({
            "name": "smote_oversampling",
            "info": smote_info,
        })
    
    X_train, X_test, y_train, y_test = split_dataset(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    
    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
    }
    preprocessing_info["steps"].append({
        "name": "train_test_split",
        "info": split_info,
    })
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "preprocessing_info": preprocessing_info,
    }


def get_dataset_statistics(df: pd.DataFrame, target_column: str = 'Class') -> Dict[str, Any]:
    """
    获取数据集统计信息。
    
    Args:
        df: 数据框
        target_column: 目标列名
        
    Returns:
        Dict: 统计信息字典
    """
    stats = {
        "basic_info": {
            "total_samples": len(df),
            "total_features": len(df.columns) - 1,
            "feature_names": [col for col in df.columns if col != target_column],
        },
        "class_distribution": {
            "normal_count": int((df[target_column] == 0).sum()),
            "fraud_count": int((df[target_column] == 1).sum()),
            "normal_ratio": float((df[target_column] == 0).sum() / len(df) * 100),
            "fraud_ratio": float((df[target_column] == 1).sum() / len(df) * 100),
        },
        "feature_statistics": {},
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats["feature_statistics"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
    
    return stats


def preprocess_german_credit(
    df: pd.DataFrame,
    categorical_cols: list,
    numerical_cols: list,
    target_col: str = "class",
    encoding_method: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    German Credit数据集预处理流程。
    
    Args:
        df: 原始数据框
        categorical_cols: 类别特征列名列表
        numerical_cols: 数值特征列名列表
        target_col: 目标列名
        encoding_method: 类别编码方法，'label' 或 'onehot'
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test, preprocessors
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import pickle
    
    df_processed = df.copy()
    
    df_processed[target_col] = df_processed[target_col] - 1
    
    preprocessors = {}
    
    if encoding_method == "label":
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
        preprocessors["label_encoders"] = label_encoders
    elif encoding_method == "onehot":
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_array = onehot_encoder.fit_transform(df_processed[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=onehot_encoder.get_feature_names_out(categorical_cols),
            index=df_processed.index
        )
        df_processed = pd.concat([
            df_processed.drop(columns=categorical_cols),
            encoded_df
        ], axis=1)
        preprocessors["onehot_encoder"] = onehot_encoder
    else:
        raise ValueError(f"不支持的编码方法: {encoding_method}")
    
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    preprocessors["scaler"] = scaler
    
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    preprocessors["feature_columns"] = list(X.columns)
    preprocessors["categorical_cols"] = categorical_cols
    preprocessors["numerical_cols"] = numerical_cols
    preprocessors["encoding_method"] = encoding_method
    
    return X_train, X_test, y_train, y_test, preprocessors


def save_preprocessors(preprocessors: Dict, save_path: str) -> None:
    """
    保存预处理器到文件。
    
    Args:
        preprocessors: 预处理器字典
        save_path: 保存路径
    """
    from pathlib import Path
    import pickle
    
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(preprocessors, f)
    print(f"预处理器已保存至: {save_path}")


def load_preprocessors(load_path: str) -> Dict:
    """
    从文件加载预处理器。
    
    Args:
        load_path: 文件路径
        
    Returns:
        Dict: 预处理器字典
    """
    import pickle
    
    with open(load_path, "rb") as f:
        preprocessors = pickle.load(f)
    return preprocessors


def get_german_credit_statistics(df: pd.DataFrame, target_col: str = "class") -> Dict:
    """
    获取German Credit数据集统计信息。
    
    Args:
        df: 数据框
        target_col: 目标列名
        
    Returns:
        Dict: 统计信息字典
    """
    stats = {
        "total_samples": len(df),
        "n_features": len(df.columns) - 1,
        "feature_names": list(df.columns[:-1]),
        "target_distribution": df[target_col].value_counts().to_dict(),
        "target_ratio": df[target_col].value_counts(normalize=True).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numerical_stats": df.describe().to_dict(),
    }
    return stats


def generate_german_credit_report(
    df: pd.DataFrame,
    categorical_cols: list,
    numerical_cols: list,
    target_col: str = "class",
    encoding_method: str = "label",
) -> str:
    """
    生成German Credit数据集统计报告。
    
    Args:
        df: 数据框
        categorical_cols: 类别特征列名列表
        numerical_cols: 数值特征列名列表
        target_col: 目标列名
        encoding_method: 编码方法
        
    Returns:
        str: Markdown格式的统计报告
    """
    stats = get_german_credit_statistics(df, target_col)
    
    report = f"""# German Credit 数据集统计报告

## 1. 数据集规模

- **总样本数**: {stats["total_samples"]}
- **特征数量**: {stats["n_features"]}
- **目标变量**: {target_col}

## 2. 特征列表及类型

### 数值特征 ({len(numerical_cols)}个)
"""
    
    for col in numerical_cols:
        col_stats = df[col].describe()
        report += f"- **{col}**: 均值={col_stats['mean']:.2f}, 标准差={col_stats['std']:.2f}, 范围=[{col_stats['min']:.2f}, {col_stats['max']:.2f}]\n"
    
    report += f"""
### 类别特征 ({len(categorical_cols)}个)
"""
    
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        report += f"- **{col}**: {unique_vals}个唯一值\n"
    
    report += f"""
## 3. 类别分布

| 类别 | 数量 | 比例 |
|------|------|------|
"""
    
    for label, count in stats["target_distribution"].items():
        label_name = "好客户" if label == 1 else "坏客户"
        ratio = stats["target_ratio"][label] * 100
        report += f"| {label_name} ({label}) | {count} | {ratio:.2f}% |\n"
    
    report += f"""
## 4. 预处理步骤说明

### 4.1 类别特征编码
- **编码方法**: {"Label Encoding" if encoding_method == "label" else "One-Hot Encoding"}
- **处理特征**: {len(categorical_cols)}个类别特征
- **说明**: {"将类别值转换为整数标签" if encoding_method == "label" else "将类别值转换为二进制向量"}

### 4.2 数值特征标准化
- **标准化方法**: StandardScaler (Z-score标准化)
- **处理特征**: {len(numerical_cols)}个数值特征
- **公式**: (x - mean) / std

### 4.3 数据集划分
- **训练集比例**: 80%
- **测试集比例**: 20%
- **划分方式**: 分层抽样 (保持类别比例)

## 5. 缺失值统计

"""
    
    missing = stats["missing_values"]
    has_missing = any(v > 0 for v in missing.values())
    
    if has_missing:
        for col, count in missing.items():
            if count > 0:
                report += f"- **{col}**: {count}个缺失值\n"
    else:
        report += "数据集无缺失值。\n"
    
    return report
