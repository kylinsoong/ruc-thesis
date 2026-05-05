import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f'原始数据:')
print(f'  X_train: {X_train.shape}')
print(f'  X_test: {X_test.shape}')

# 10%采样 (stratified) - 使用frac参数
test_sample_ratio = 0.1  # 保留原测试集的10%
_, X_test_small, _, y_test_small = train_test_split(
    X_test, y_test, test_size=test_sample_ratio, stratify=y_test, random_state=42
)

_, X_train_small, _, y_train_small = train_test_split(
    X_train, y_train, test_size=test_sample_ratio, stratify=y_train, random_state=42
)

print(f'\n10%采样后:')
print(f'  X_train_small: {X_train_small.shape}')
print(f'  X_test_small: {X_test_small.shape}')
print(f'  y_train_small: {y_train_small.shape}')
print(f'  y_test_small: {y_test_small.shape}')

# 保存
X_train_small.to_csv('data/processed/credit_card_llm_X_train.csv', index=False)
X_test_small.to_csv('data/processed/credit_card_llm_X_test.csv', index=False)
y_train_small.to_csv('data/processed/credit_card_llm_y_train.csv', index=False)
y_test_small.to_csv('data/processed/credit_card_llm_y_test.csv', index=False)

print('\n已保存为 credit_card_llm 数据集:')
print('  credit_card_llm_X_train.csv')
print('  credit_card_llm_X_test.csv')
print('  credit_card_llm_y_train.csv')
print('  credit_card_llm_y_test.csv')
