"""
数据处理主脚本

整合数据下载、预处理和统计报告生成功能。
"""

import os
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import DATA_CONFIG, EXPERIMENT_CONFIG
from utils.data_loader import (
    download_credit_card_fraud_data,
    load_credit_card_data,
    verify_data_integrity,
    get_manual_download_instructions,
)
from utils.preprocess import (
    preprocess_credit_card_data,
    get_dataset_statistics,
)


def create_mock_data(save_path: str, n_samples: int = 10000) -> str:
    """
    创建模拟数据用于测试。
    
    Args:
        save_path: 保存路径
        n_samples: 样本数量
        
    Returns:
        str: 数据文件路径
    """
    np.random.seed(42)
    
    fraud_ratio = 0.00172
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    data = {
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.random.exponential(88.35, n_samples),
    }
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    class_col = np.array([0] * n_normal + [1] * n_fraud)
    np.random.shuffle(class_col)
    data['Class'] = class_col
    
    df = pd.DataFrame(data)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"模拟数据已创建: {save_path}")
    print(f"  - 总样本数: {len(df)}")
    print(f"  - 欺诈样本数: {df['Class'].sum()}")
    
    return save_path


def generate_stats_report(
    original_stats: dict,
    preprocessing_info: dict,
    output_path: str,
) -> None:
    """
    生成数据集统计报告。
    
    Args:
        original_stats: 原始数据统计信息
        preprocessing_info: 预处理信息
        output_path: 输出文件路径
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    basic_info = original_stats["basic_info"]
    class_dist = original_stats["class_distribution"]
    
    report = f"""# Credit Card Fraud 数据集统计报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 数据集概述

### 1.1 基本信息

| 指标 | 数值 |
|------|------|
| 总样本数 | {basic_info['total_samples']:,} |
| 特征数量 | {basic_info['total_features']} |
| 目标变量 | Class (0=正常, 1=欺诈) |

### 1.2 数据来源

- **数据集名称**: Credit Card Fraud Detection
- **来源**: Kaggle
- **URL**: https://www.kaggle.com/mlg-ulb/creditcardfraud
- **描述**: 该数据集包含2013年9月欧洲持卡人的信用卡交易数据

## 2. 类别分布

| 类别 | 数量 | 比例 |
|------|------|------|
| 正常交易 (Class=0) | {class_dist['normal_count']:,} | {class_dist['normal_ratio']:.4f}% |
| 欺诈交易 (Class=1) | {class_dist['fraud_count']:,} | {class_dist['fraud_ratio']:.4f}% |

**数据不平衡程度**: 欺诈交易仅占 {class_dist['fraud_ratio']:.4f}%，数据集高度不平衡。

## 3. 特征说明

### 3.1 特征列表

数据集包含以下特征：

| 特征名 | 说明 |
|--------|------|
| Time | 每笔交易与第一笔交易的时间差（秒） |
| Amount | 交易金额 |
| V1-V28 | PCA转换后的主成分特征（出于隐私保护，原始特征未公开） |
| Class | 目标变量（0=正常，1=欺诈） |

### 3.2 关键特征统计

#### Time 特征

"""
    
    if 'Time' in original_stats['feature_statistics']:
        time_stats = original_stats['feature_statistics']['Time']
        report += f"""| 统计量 | 数值 |
|--------|------|
| 均值 | {time_stats['mean']:.2f} 秒 |
| 标准差 | {time_stats['std']:.2f} 秒 |
| 最小值 | {time_stats['min']:.2f} 秒 |
| 最大值 | {time_stats['max']:.2f} 秒 |
| 中位数 | {time_stats['median']:.2f} 秒 |

"""
    
    report += """#### Amount 特征

"""
    
    if 'Amount' in original_stats['feature_statistics']:
        amount_stats = original_stats['feature_statistics']['Amount']
        report += f"""| 统计量 | 数值 |
|--------|------|
| 均值 | ${amount_stats['mean']:.2f} |
| 标准差 | ${amount_stats['std']:.2f} |
| 最小值 | ${amount_stats['min']:.2f} |
| 最大值 | ${amount_stats['max']:.2f} |
| 中位数 | ${amount_stats['median']:.2f} |

"""
    
    report += """## 4. 数据预处理步骤

"""
    
    for i, step in enumerate(preprocessing_info.get('steps', []), 1):
        step_name = step['name']
        step_info = step['info']
        
        if step_name == 'standardization':
            report += f"""### 4.{i} 特征标准化

对以下特征进行标准化处理（Z-score标准化）：

- 标准化特征: {', '.join(step_info['columns'])}
- 标准化公式: (x - mean) / std

"""
        elif step_name == 'smote_oversampling':
            orig_dist = step_info['original_distribution']
            resamp_dist = step_info['resampled_distribution']
            report += f"""### 4.{i} SMOTE过采样

处理类别不平衡问题：

| 指标 | 处理前 | 处理后 |
|------|--------|--------|
| 正常样本数 | {orig_dist.get(0, 0):,} | {resamp_dist.get(0, 0):,} |
| 欺诈样本数 | {orig_dist.get(1, 0):,} | {resamp_dist.get(1, 0):,} |
| 总样本数 | {step_info['original_samples']:,} | {step_info['resampled_samples']:,} |

新增样本数: {step_info['samples_added']:,}

"""
        elif step_name == 'train_test_split':
            report += f"""### 4.{i} 数据集划分

采用分层抽样进行训练集/测试集划分：

| 数据集 | 样本数 | 正常样本 | 欺诈样本 |
|--------|--------|----------|----------|
| 训练集 | {step_info['train_size']:,} | {step_info['train_class_distribution'].get(0, 0):,} | {step_info['train_class_distribution'].get(1, 0):,} |
| 测试集 | {step_info['test_size']:,} | {step_info['test_class_distribution'].get(0, 0):,} | {step_info['test_class_distribution'].get(1, 0):,} |

划分比例: 训练集 80%, 测试集 20%

"""
    
    report += """## 5. 数据质量检查

"""
    
    missing_count = preprocessing_info.get('missing_report', {}).get('missing_count', {}).get('column', 0)
    if isinstance(missing_count, list):
        total_missing = sum(preprocessing_info.get('missing_report', {}).get('missing_count', {}).get('missing_count', [0]))
    else:
        total_missing = 0
    
    report += f"""| 检查项 | 结果 |
|--------|------|
| 缺失值 | {'无缺失值' if total_missing == 0 else f'存在缺失值，已处理'} |
| 重复记录 | 建议在分析时检查 |
| 数据类型 | 全部为数值型 |

## 6. 使用建议

### 6.1 模型选择建议

由于数据集高度不平衡，建议：

1. 使用适合不平衡数据的评估指标（如AUC-ROC、F1-score、Precision-Recall曲线）
2. 考虑使用代价敏感学习
3. 可尝试其他采样方法（如ADASYN、欠采样等）

### 6.2 特征工程建议

1. Time特征可考虑转换为周期性特征（如一天中的时间段）
2. Amount特征可考虑对数变换
3. 可探索V1-V28特征之间的交互

## 7. 参考资料

- 数据集原始论文: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. "Calibrating Probability with Undersampling for Unbalanced Classification." In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
- Kaggle数据集页面: https://www.kaggle.com/mlg-ulb/creditcardfraud
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"统计报告已保存到: {output_path}")


def save_processed_data(
    processed_data: dict,
    output_dir: str,
) -> None:
    """
    保存处理后的数据。
    
    Args:
        processed_data: 处理后的数据字典
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_data['X_train'].to_csv(output_path / 'X_train.csv', index=False)
    processed_data['X_test'].to_csv(output_path / 'X_test.csv', index=False)
    processed_data['y_train'].to_csv(output_path / 'y_train.csv', index=False, header=['Class'])
    processed_data['y_test'].to_csv(output_path / 'y_test.csv', index=False, header=['Class'])
    
    preprocessing_info = processed_data['preprocessing_info'].copy()
    for step in preprocessing_info.get('steps', []):
        if 'info' in step and isinstance(step['info'], dict):
            for key, value in step['info'].items():
                if hasattr(value, 'to_dict'):
                    step['info'][key] = value.to_dict()
    
    with open(output_path / 'preprocessing_info.json', 'w', encoding='utf-8') as f:
        json.dump(preprocessing_info, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"处理后的数据已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Credit Card Fraud数据集准备')
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='跳过下载步骤（使用已有数据）',
    )
    parser.add_argument(
        '--skip-smote',
        action='store_true',
        help='跳过SMOTE过采样',
    )
    parser.add_argument(
        '--use-mock-data',
        action='store_true',
        help='使用模拟数据进行测试',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=EXPERIMENT_CONFIG['random_seed'],
        help='随机种子',
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / DATA_CONFIG['raw_data_dir']
    processed_data_dir = project_root / DATA_CONFIG['processed_data_dir']
    stats_dir = project_root / 'results' / 'data_stats'
    
    print("=" * 60)
    print("Credit Card Fraud 数据集准备流程")
    print("=" * 60)
    
    print("\n[步骤 1] 数据下载")
    print("-" * 40)
    
    data_file = raw_data_dir / 'creditcard.csv'
    
    if args.use_mock_data:
        print("使用模拟数据进行测试...")
        create_mock_data(str(data_file), n_samples=10000)
    elif not args.skip_download:
        try:
            downloaded_file = download_credit_card_fraud_data(str(raw_data_dir))
            if downloaded_file:
                data_file = Path(downloaded_file)
        except Exception as e:
            print(f"自动下载失败: {e}")
            print(get_manual_download_instructions())
    
    if not data_file.exists():
        print(f"\n错误: 数据文件不存在: {data_file}")
        print(get_manual_download_instructions())
        return
    
    print("\n[步骤 2] 数据加载与验证")
    print("-" * 40)
    
    df = load_credit_card_data(str(data_file))
    
    verification = verify_data_integrity(df)
    print(f"数据验证结果:")
    print(f"  - 行数: {verification['row_count']:,}")
    print(f"  - 列数: {verification['column_count']}")
    print(f"  - 包含所有预期列: {verification['has_all_columns']}")
    print(f"  - 缺失值总数: {verification['missing_values']}")
    print(f"  - 重复行数: {verification['duplicate_rows']}")
    
    print("\n[步骤 3] 数据预处理")
    print("-" * 40)
    
    processed_data = preprocess_credit_card_data(
        df,
        target_column='Class',
        standardize_cols=['Amount', 'Time'],
        apply_smote_oversampling=not args.skip_smote,
        test_size=0.2,
        random_state=args.random_seed,
    )
    
    print(f"预处理完成:")
    print(f"  - 训练集大小: {len(processed_data['X_train']):,}")
    print(f"  - 测试集大小: {len(processed_data['X_test']):,}")
    print(f"  - 特征数量: {len(processed_data['feature_columns'])}")
    
    print("\n[步骤 4] 生成统计报告")
    print("-" * 40)
    
    original_stats = get_dataset_statistics(df)
    stats_file = stats_dir / 'credit_card_stats.md'
    generate_stats_report(original_stats, processed_data['preprocessing_info'], str(stats_file))
    
    print("\n[步骤 5] 保存处理后的数据")
    print("-" * 40)
    save_processed_data(processed_data, str(processed_data_dir))
    
    print("\n" + "=" * 60)
    print("数据准备流程完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 统计报告: {stats_file}")
    print(f"  - 处理后数据: {processed_data_dir}/")


if __name__ == '__main__':
    main()
