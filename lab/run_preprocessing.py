#!/usr/bin/env python3
# run_preprocessing.py - 执行数据预处理

import sys
sys.path.insert(0, '.')

from data.credit_card import loader as cc_loader
from data.credit_card import preprocessing as cc_prep
from data.german_credit import loader as gc_loader
from data.german_credit import preprocessing as gc_prep
from data.knowledge_base import builder as kb_builder

def main():
    print("=" * 50)
    print("Credit Card 数据预处理")
    print("=" * 50)
    cc_df = cc_loader.CreditCardLoader.load(sample_size=10000)
    cc_processor = cc_prep.CreditCardPreprocessor()
    cc_result = cc_processor.preprocess(cc_df)
    print(f"训练集: {len(cc_result['X_train'])} 条")
    print(f"测试集: {len(cc_result['X_test'])} 条")

    print()
    print("=" * 50)
    print("German Credit 数据预处理")
    print("=" * 50)
    gc_df = gc_loader.GermanCreditLoader.load()
    gc_processor = gc_prep.GermanCreditPreprocessor()
    gc_result = gc_processor.preprocess(gc_df)
    print(f"训练集: {len(gc_result['X_train'])} 条")
    print(f"测试集: {len(gc_result['X_test'])} 条")

    print()
    print("=" * 50)
    print("知识库构建")
    print("=" * 50)
    kb = kb_builder.KnowledgeBaseBuilder()
    kb_result = kb.build()
    print(f"知识文档数量: {kb_result['count']}")

    print()
    print("=" * 50)
    print("预处理完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
