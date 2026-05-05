import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_loader import get_german_credit, get_feature_info
from utils.preprocess import (
    preprocess_german_credit,
    generate_german_credit_report,
    save_preprocessors,
)


def main():
    project_dir = Path(__file__).parent.parent
    
    print("=" * 60)
    print("German Credit 数据集构建流程")
    print("=" * 60)
    
    print("\n步骤1: 下载German Credit数据集")
    print("-" * 40)
    save_dir = project_dir / "data" / "raw"
    df = get_german_credit(str(save_dir))
    print(f"数据集加载完成，共 {len(df)} 条记录")
    
    print("\n步骤2: 获取特征信息")
    print("-" * 40)
    feature_info = get_feature_info()
    print(f"类别特征数量: {len(feature_info['categorical'])}")
    print(f"数值特征数量: {len(feature_info['numerical'])}")
    
    print("\n步骤3: 数据预处理")
    print("-" * 40)
    X_train, X_test, y_train, y_test, preprocessors = preprocess_german_credit(
        df=df,
        categorical_cols=feature_info["categorical"],
        numerical_cols=feature_info["numerical"],
        target_col=feature_info["target"],
        encoding_method="label",
        test_size=0.2,
        random_state=42,
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集好客户比例: {(y_train == 0).sum() / len(y_train) * 100:.2f}%")
    print(f"测试集好客户比例: {(y_test == 0).sum() / len(y_test) * 100:.2f}%")
    
    print("\n步骤4: 保存预处理器")
    print("-" * 40)
    preprocessor_path = project_dir / "data" / "processed" / "preprocessors.pkl"
    save_preprocessors(preprocessors, str(preprocessor_path))
    
    print("\n步骤5: 生成统计报告")
    print("-" * 40)
    report = generate_german_credit_report(
        df=df,
        categorical_cols=feature_info["categorical"],
        numerical_cols=feature_info["numerical"],
        target_col=feature_info["target"],
        encoding_method="label",
    )
    
    stats_dir = project_dir / "results" / "data_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    report_path = stats_dir / "german_credit_stats.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"统计报告已保存至: {report_path}")
    
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)
    
    return df, X_train, X_test, y_train, y_test, preprocessors


if __name__ == "__main__":
    main()
