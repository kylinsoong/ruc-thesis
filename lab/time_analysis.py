#!/usr/bin/env python
import sys
import time
sys.path.insert(0, '.')

from data_loader import load_data
from models.factory import create_model
import torch
torch.set_num_threads(1)

datasets = ['german_credit', 'credit_card']
models = ['lr', 'rf', 'xgboost', 'deepfm', 'tabnet', 'llm']

print('=' * 60)
print('基准模型实验时间分析')
print('=' * 60)

total_time = {}

for dataset in datasets:
    print(f'\n数据集: {dataset}')
    print('-' * 40)

    X_train, X_test, y_train, y_test = load_data(dataset)
    print(f'训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本')
    print()

    for model_name in models:
        if model_name == 'llm':
            model = create_model(model_name, mock_mode=True, random_state=42)
        elif model_name == 'xgboost':
            model = create_model(model_name, random_state=42)
        else:
            model = create_model(model_name, random_state=42)

        start = time.time()
        if hasattr(model, 'train'):
            model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        elapsed = time.time() - start

        key = f'{model_name}_{dataset}'
        total_time[key] = elapsed

        print(f'{model_name:12s}: {elapsed:6.2f}秒 | F1={metrics["f1"]:.4f}')

print('\n' + '=' * 60)
print('时间汇总')
print('=' * 60)

print('\n各模型总时间（2个数据集）:')
for model_name in models:
    t = total_time.get(f'{model_name}_german_credit', 0) + total_time.get(f'{model_name}_credit_card', 0)
    print(f'{model_name:12s}: {t:6.2f}秒')

print(f'\n预计总时间: {sum(total_time.values()):.2f}秒')
