#!/usr/bin/env python
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from models.baseline.llm_only import LLMOnlyModel


def main():
    parser = argparse.ArgumentParser(description="Test LLM on a single sample from Credit Card dataset")
    parser.add_argument("--index", type=int, default=0, help="Sample index to test (default: 0)")
    args = parser.parse_args()

    print("=" * 60)
    print("LLM 单条数据测试 - Credit Card Fraud 数据集")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data("credit_card")
    print(f"\n数据集信息:")
    print(f"  训练集大小: {len(X_train)}")
    print(f"  测试集大小: {len(X_test)}")
    print(f"  正样本数(欺诈): {int(sum(y_test))}")
    print(f"  负样本数(正常): {int(len(y_test) - sum(y_test))}")

    if args.index >= len(X_test):
        print(f"\n错误: 索引 {args.index} 超出测试集范围 (0-{len(X_test)-1})")
        return

    sample = X_test.iloc[args.index]
    true_label = y_test.iloc[args.index]

    print(f"\n测试样本信息:")
    print(f"  样本索引: {args.index}")
    print(f"  特征数: {len(sample)}")
    print(f"  真实标签: {true_label} ({'欺诈' if true_label == 1 else '正常'})")

    print("\n" + "=" * 60)
    print("创建 LLM 模型...")
    print("=" * 60)

    model = LLMOnlyModel(mock_mode=False)

    print("\n" + "=" * 60)
    print("生成的 Prompt:")
    print("=" * 60)
    prompt = model._build_prompt(sample.to_dict(), list(sample.index))
    print(prompt)

    print("\n" + "=" * 60)
    print("调用 LLM API 进行预测...")
    print("=" * 60)

    risk_score, pred_label = model.predict_single(sample.to_dict())

    print("\n" + "=" * 60)
    print("预测结果:")
    print("=" * 60)
    print(f"  真实标签: {true_label} ({'欺诈' if true_label == 1 else '正常'})")
    print(f"  预测标签: {pred_label} ({'欺诈' if pred_label == 1 else '正常'})")
    print(f"  风险评分: {risk_score:.4f} (0=低风险, 1=高风险)")
    print(f"  预测结果: {'正确' if true_label == pred_label else '错误'}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
