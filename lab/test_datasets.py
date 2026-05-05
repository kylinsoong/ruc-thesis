#!/usr/bin/env python3
"""
数据集测试代码
用于查看和验证数据集状态
"""
import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def test_credit_card():
    """测试 Credit Card 数据集"""
    print("=" * 60)
    print("Credit Card Fraud 数据集")
    print("=" * 60)

    # 加载处理后的数据
    X_train = pd.read_csv(DATA_DIR / "processed" / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "processed" / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "processed" / "y_train.csv")
    y_test = pd.read_csv(DATA_DIR / "processed" / "y_test.csv")

    print(f"\n训练集 X_train: {X_train.shape}")
    print(f"测试集 X_test: {X_test.shape}")
    print(f"训练集 y_train: {y_train.shape}")
    print(f"测试集 y_test: {y_test.shape}")

    print(f"\n特征数量: {X_train.shape[1]}")
    print(f"特征列表: {list(X_train.columns[:10])}...")

    print(f"\n训练集类别分布:")
    print(y_train.value_counts())

    print(f"\n测试集类别分布:")
    print(y_test.value_counts())

    print(f"\n前3条训练数据:")
    print(X_train.head(3))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def test_german_credit():
    """测试 German Credit 数据集"""
    print("\n" + "=" * 60)
    print("German Credit 数据集")
    print("=" * 60)

    # 加载处理后的数据
    X_train = pd.read_csv(DATA_DIR / "processed" / "german_X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "processed" / "german_X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "processed" / "german_y_train.csv")
    y_test = pd.read_csv(DATA_DIR / "processed" / "german_y_test.csv")

    print(f"\n训练集 X_train: {X_train.shape}")
    print(f"测试集 X_test: {X_test.shape}")
    print(f"训练集 y_train: {y_train.shape}")
    print(f"测试集 y_test: {y_test.shape}")

    print(f"\n特征数量: {X_train.shape[1]}")
    print(f"特征列表: {list(X_train.columns)}")

    print(f"\n训练集类别分布 (0=好, 1=坏):")
    print(y_train["class"].value_counts())

    print(f"\n测试集类别分布 (0=好, 1=坏):")
    print(y_test["class"].value_counts())

    print(f"\n前3条训练数据:")
    print(X_train.head(3))

    # 读取特征说明
    feature_desc = DATA_DIR / "feature_description.md"
    if feature_desc.exists():
        print(f"\n特征说明文件: {feature_desc}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def test_raw_data():
    """查看原始数据信息"""
    print("\n" + "=" * 60)
    print("原始数据文件")
    print("=" * 60)

    raw_dir = DATA_DIR / "raw"
    if raw_dir.exists():
        for f in raw_dir.iterdir():
            size = f.stat().st_size / 1024
            print(f"  {f.name}: {size:.1f} KB")

def test_preprocessing_info():
    """查看预处理信息"""
    print("\n" + "=" * 60)
    print("预处理信息")
    print("=" * 60)

    info_file = DATA_DIR / "processed" / "preprocessing_info.json"
    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)

        print(f"\n原始数据形状: {info['original_shape']}")
        print(f"\n处理步骤:")
        for step in info['steps']:
            print(f"  - {step['name']}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("数据集状态检查")
    print("=" * 60)

    test_credit_card()
    test_german_credit()
    test_raw_data()
    test_preprocessing_info()

    print("\n" + "=" * 60)
    print("数据集检查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
