#!/usr/bin/env python
"""测试模型加载和预测"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from models.baseline.deep_learning import DeepFMModel, TabNetModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("=" * 60)
print("测试 DeepFM 模型")
print("=" * 60)

# 加载数据
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# 加载DeepFM模型
print("加载模型...")
model = DeepFMModel(batch_size=256, random_state=42)
model.load('results/models/deepfm_credit_card.pt')
print(f"模型加载完成，is_fitted: {model.is_fitted}")

# 预测
print('开始预测...')
y_pred = model.predict(X_test)
print(f'预测完成，y_pred shape: {y_pred.shape}')

y_prob = model.predict_proba(X_test)
print(f'概率预测完成，y_prob shape: {y_prob.shape}')

# 计算指标
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'F1: {f1_score(y_test, y_pred):.4f}')
print(f'AUC-ROC: {roc_auc_score(y_test, y_prob[:, 1]):.4f}')

print("\n" + "=" * 60)
print("测试 TabNet 模型")
print("=" * 60)

# 加载TabNet模型
model2 = TabNetModel(random_state=42)
model2.load('results/models/tabnet_credit_card.pt.zip')

# 预测
y_pred2 = model2.predict(X_test)
y_prob2 = model2.predict_proba(X_test)

print(f'y_pred shape: {y_pred2.shape}')
print(f'y_prob shape: {y_prob2.shape}')

# 计算指标
print(f'Accuracy: {accuracy_score(y_test, y_pred2):.4f}')
print(f'F1: {f1_score(y_test, y_pred2):.4f}')
print(f'AUC-ROC: {roc_auc_score(y_test, y_prob2[:, 1]):.4f}')

print("\n测试完成!")
