# 基线模型实验运行指南

本文档说明如何在 `/Users/bytedance/src/coding.ai/ruc/lab/` 目录下运行6大基线模型实验。

## 1. 环境准备

```bash
cd /Users/bytedance/src/coding.ai/ruc/lab
pip install -r requirements.txt
```

确保已配置 `.env` 文件（参考 `.env.example`）。

## 2. 支持的模型

| 模型名称 | 模型类型 | 运行标识 |
|----------|----------|----------|
| Logistic Regression | 传统机器学习 | `lr` |
| Random Forest | 传统机器学习 | `rf` |
| XGBoost | 传统机器学习 | `xgboost` |
| DeepFM | 深度学习 | `deepfm` |
| TabNet | 深度学习 | `tabnet` |
| LLM-only | 大语言模型 | `llm` |

## 3. 支持的数据集

| 数据集名称 | 运行标识 |
|------------|----------|
| German Credit | `german_credit` |
| Credit Card Fraud | `credit_card` |

## 4. 单个模型运行

### 4.1 运行指定模型在指定数据集

```bash
# 语法
python run_baseline_experiment.py --model <模型名> --dataset <数据集名>

# 示例：运行LR模型在German Credit数据集
python run_baseline_experiment.py --model lr --dataset german_credit

# 示例：运行XGBoost模型在Credit Card数据集
python run_baseline_experiment.py --model xgboost --dataset credit_card
```

### 4.2 运行指定模型在所有数据集

```bash
# 使用 --all-datasets 参数
python run_baseline_experiment.py --model lr --all-datasets
```

### 4.3 使用Mock模式运行LLM

LLM模型调用真实API会产生费用。测试时可使用Mock模式：

```bash
python run_baseline_experiment.py --model llm --dataset german_credit --mock-llm
```

### 4.4 重复实验

使用 `--n-repeats` 参数设置重复次数：

```bash
python run_baseline_experiment.py --model lr --dataset german_credit --n-repeats 5
```

## 5. 批量运行所有模型

### 5.1 运行所有模型在所有数据集

```bash
python run_baseline_experiment.py --all-models --all-datasets
```

或直接使用快捷脚本：

```bash
python run_all_experiments.py
```

### 5.2 输出结果

结果文件保存在：`lab/results/baseline_results.csv`

CSV格式：

| 列名 | 说明 |
|------|------|
| model_type | 模型名称 |
| dataset | 数据集名称 |
| accuracy | 准确率 |
| precision | 精确率 |
| recall | 召回率 |
| f1 | F1分数 |
| auc_roc | ROC曲线下面积 |
| auc_pr | PR曲线下面积 |
| train_samples | 训练集样本数 |
| test_samples | 测试集样本数 |

## 6. 实验结果示例

### German Credit数据集

| 模型 | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|------|----------|-----------|--------|-----|---------|--------|
| LR | 0.715 | 0.518 | 0.733 | 0.607 | 0.790 | 0.607 |
| RF | 0.750 | 0.656 | 0.350 | 0.457 | 0.783 | 0.586 |
| XGBoost | 0.770 | 0.659 | 0.483 | 0.558 | 0.796 | 0.643 |
| DeepFM | 0.785 | 0.667 | 0.567 | 0.613 | 0.798 | 0.655 |
| TabNet | 0.740 | 0.583 | 0.467 | 0.519 | 0.749 | 0.582 |
| LLM | 0.650 | 0.582 | 0.524 | 0.552 | 0.683 | 0.661 |

### Credit Card Fraud数据集

| 模型 | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|------|----------|-----------|--------|-----|---------|--------|
| LR | 0.855 | 0.827 | 0.897 | 0.861 | 0.895 | 0.850 |
| RF | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 |
| XGBoost | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 |
| DeepFM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TabNet | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LLM | 0.723 | 0.654 | 0.582 | 0.616 | 0.752 | 0.781 |

## 7. 命令行参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 指定模型 | `--model lr` |
| `--dataset` | 指定数据集 | `--dataset german_credit` |
| `--all-models` | 运行所有模型 | `--all-models` |
| `--all-datasets` | 运行所有数据集 | `--all-datasets` |
| `--n-repeats` | 重复次数 | `--n-repeats 5` |
| `--mock-llm` | 使用Mock LLM | `--mock-llm` |
| `--output` | 结果输出路径 | `--output results/my_results.csv` |

## 8. 常见问题

### Q: LLM模型调用报错？

A: 检查 `.env` 文件中的 `ARK_API_KEY` 是否正确配置，或使用 `--mock-llm` 参数进行测试。

### Q: DeepFM/TabNet训练很慢？

A: 这些是深度学习模型，训练时间较长。可以减少epoch数或使用更大的batch size。

### Q: 如何只查看某个模型的结果？

A: 查看 `lab/results/baseline_results.csv` 文件，使用grep过滤：

```bash
grep "lr," lab/results/baseline_results.csv
```
