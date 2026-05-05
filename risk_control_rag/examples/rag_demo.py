"""
RAG风险分析系统使用示例

展示如何使用RAG系统进行风险分析，包括：
- 基本使用方法
- 不同配置选项
- 多案例分析
- 结果解析
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag.rag_system import (
    RAGRiskSystem,
    RAGSystemConfig,
    RetrievalStrategy,
    create_rag_system,
)
from models.rag.batch_predictor import (
    BatchPredictor,
    BatchPredictorConfig,
    create_batch_predictor,
)
from models.rag.output_parser import (
    OutputParser,
    RiskLevel,
    ApprovalSuggestion,
)


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "=" * 60)
    print("示例1: 基本使用方法")
    print("=" * 60)
    
    rag_system = create_rag_system(
        retrieval_strategy="hybrid",
        top_k=5,
        temperature=0.3,
    )
    
    customer_data = {
        "checking_status": "A11",
        "duration": 24,
        "credit_history": "A32",
        "purpose": "A43",
        "credit_amount": 5000,
        "savings_status": "A61",
        "employment": "A73",
        "installment_commitment": 3,
        "personal_status": "A93",
        "other_parties": "A101",
        "residence_since": 2,
        "property_magnitude": "A121",
        "age": 35,
        "other_payment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "num_dependents": 1,
        "own_telephone": "A192",
        "foreign_worker": "A201",
    }
    
    print("\n客户信息:")
    print("  - 支票账户: 账户透支")
    print("  - 贷款期限: 24个月")
    print("  - 信用历史: 现有贷款按时还款")
    print("  - 贷款目的: 购买收音机/电视")
    print("  - 贷款金额: 5000 DM")
    print("  - 年龄: 35岁")
    
    print("\n正在分析...")
    result = rag_system.analyze(customer_data, customer_id="DEMO_001")
    
    print("\n分析结果:")
    result_dict = result.to_dict()
    print(f"  风险等级: {result_dict['risk_level']}")
    print(f"  风险评分: {result_dict['risk_score']}")
    print(f"  审批建议: {result_dict['approval_suggestion']}")
    
    if result_dict['risk_factors']:
        print("\n  主要风险因素:")
        for i, factor in enumerate(result_dict['risk_factors'][:3], 1):
            print(f"    {i}. {factor['name']}")
            print(f"       描述: {factor['description'][:60]}...")
            print(f"       权重: {factor['weight']}")
    
    if result_dict['knowledge_sources']:
        print(f"\n  引用知识来源: {len(result_dict['knowledge_sources'])} 条")
    
    print(f"\n  Token消耗: {result_dict.get('token_usage', {}).get('total_tokens', 0)}")
    print(f"  费用: ¥{result_dict.get('cost', 0):.4f}")
    print(f"  耗时: {result_dict['latency']:.2f}秒")
    
    return result


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("示例2: 自定义配置")
    print("=" * 60)
    
    config = RAGSystemConfig(
        retrieval_strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        similarity_threshold=0.6,
        temperature=0.2,
        max_tokens=1024,
        template_type="concise",
        max_knowledge_items=3,
        fusion_method="weighted",
        vector_weight=0.6,
        bm25_weight=0.4,
    )
    
    print("\n自定义配置:")
    print(f"  检索策略: {config.retrieval_strategy.value}")
    print(f"  Top-K: {config.top_k}")
    print(f"  相似度阈值: {config.similarity_threshold}")
    print(f"  温度: {config.temperature}")
    print(f"  最大Token: {config.max_tokens}")
    print(f"  模板类型: {config.template_type}")
    print(f"  融合方法: {config.fusion_method}")
    
    rag_system = RAGRiskSystem(config=config)
    
    customer_data = {
        "checking_status": "A14",
        "duration": 12,
        "credit_history": "A31",
        "purpose": "A40",
        "credit_amount": 3000,
        "savings_status": "A64",
        "employment": "A75",
        "installment_commitment": 2,
        "personal_status": "A94",
        "other_parties": "A101",
        "residence_since": 4,
        "property_magnitude": "A121",
        "age": 45,
        "other_payment_plans": "A143",
        "housing": "A152",
        "existing_credits": 2,
        "job": "A174",
        "num_dependents": 2,
        "own_telephone": "A192",
        "foreign_worker": "A202",
    }
    
    print("\n客户信息:")
    print("  - 支票账户: 无支票账户")
    print("  - 贷款期限: 12个月")
    print("  - 信用历史: 所有贷款已按时还清")
    print("  - 贷款目的: 购买新车")
    print("  - 贷款金额: 3000 DM")
    print("  - 年龄: 45岁")
    
    print("\n正在分析...")
    result = rag_system.analyze(customer_data, customer_id="DEMO_002")
    
    result_dict = result.to_dict()
    print(f"\n分析结果:")
    print(f"  风险等级: {result_dict['risk_level']}")
    print(f"  风险评分: {result_dict['risk_score']}")
    print(f"  审批建议: {result_dict['approval_suggestion']}")
    
    return result


def example_batch_prediction():
    """批量预测示例"""
    print("\n" + "=" * 60)
    print("示例3: 批量预测")
    print("=" * 60)
    
    rag_system = create_rag_system()
    
    batch_data = [
        {
            "checking_status": "A11",
            "duration": 36,
            "credit_history": "A33",
            "purpose": "A41",
            "credit_amount": 12000,
            "savings_status": "A61",
            "employment": "A72",
            "age": 28,
        },
        {
            "checking_status": "A14",
            "duration": 12,
            "credit_history": "A31",
            "purpose": "A40",
            "credit_amount": 3000,
            "savings_status": "A64",
            "employment": "A75",
            "age": 45,
        },
        {
            "checking_status": "A12",
            "duration": 24,
            "credit_history": "A32",
            "purpose": "A42",
            "credit_amount": 6000,
            "savings_status": "A62",
            "employment": "A73",
            "age": 38,
        },
    ]
    
    customer_ids = ["HIGH_RISK", "LOW_RISK", "MEDIUM_RISK"]
    
    print(f"\n批量预测 {len(batch_data)} 条数据...")
    
    def progress_callback(current: int, total: int, message: str):
        print(f"  进度: {current}/{total} - {message}")
    
    predictor = create_batch_predictor(
        rag_system=rag_system,
        cache_enabled=True,
        progress_callback=progress_callback,
    )
    
    batch_result = predictor.predict_batch(
        batch_data=batch_data,
        customer_ids=customer_ids,
    )
    
    print("\n批量预测结果:")
    print(f"  总数: {batch_result.total_count}")
    print(f"  成功: {batch_result.success_count}")
    print(f"  失败: {batch_result.failed_count}")
    print(f"  总Token: {batch_result.total_tokens}")
    print(f"  总费用: ¥{batch_result.total_cost:.4f}")
    
    print("\n各样本结果:")
    for result in batch_result.results:
        print(f"\n  [{result['customer_id']}]")
        print(f"    风险等级: {result['risk_level']}")
        print(f"    风险评分: {result['risk_score']}")
        print(f"    审批建议: {result['approval_suggestion']}")
    
    return batch_result


def example_multiple_cases():
    """多案例分析示例"""
    print("\n" + "=" * 60)
    print("示例4: 多案例分析")
    print("=" * 60)
    
    rag_system = create_rag_system(
        retrieval_strategy="hybrid",
        top_k=5,
    )
    
    cases = [
        {
            "name": "高风险案例",
            "data": {
                "checking_status": "A11",
                "duration": 48,
                "credit_history": "A34",
                "purpose": "A46",
                "credit_amount": 15000,
                "savings_status": "A61",
                "employment": "A71",
                "age": 22,
            },
        },
        {
            "name": "低风险案例",
            "data": {
                "checking_status": "A14",
                "duration": 6,
                "credit_history": "A31",
                "purpose": "A40",
                "credit_amount": 2000,
                "savings_status": "A64",
                "employment": "A75",
                "age": 50,
            },
        },
        {
            "name": "中风险案例",
            "data": {
                "checking_status": "A12",
                "duration": 24,
                "credit_history": "A32",
                "purpose": "A42",
                "credit_amount": 8000,
                "savings_status": "A62",
                "employment": "A73",
                "age": 35,
            },
        },
    ]
    
    results = []
    
    for case in cases:
        print(f"\n分析 {case['name']}...")
        result = rag_system.analyze(case['data'])
        result_dict = result.to_dict()
        
        print(f"  风险等级: {result_dict['risk_level']}")
        print(f"  风险评分: {result_dict['risk_score']}")
        print(f"  审批建议: {result_dict['approval_suggestion']}")
        
        results.append({
            "case_name": case['name'],
            "result": result_dict,
        })
    
    print("\n案例分析对比:")
    print(f"\n{'案例名称':<15} {'风险等级':<10} {'风险评分':<10} {'审批建议':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['case_name']:<15} {r['result']['risk_level']:<10} {r['result']['risk_score']:<10} {r['result']['approval_suggestion']:<12}")
    
    return results


def example_output_parsing():
    """输出解析示例"""
    print("\n" + "=" * 60)
    print("示例5: 输出解析")
    print("=" * 60)
    
    sample_response = """
## 风险等级判定
**高风险**

## 关键风险因素分析

1. **账户透支风险**
   - 风险描述: 申请人支票账户处于透支状态，表明财务状况紧张
   - 风险权重: 高
   - 参考来源: [1]

2. **高额贷款风险**
   - 风险描述: 申请贷款金额较大，还款压力较大
   - 风险权重: 中

3. **就业不稳定风险**
   - 风险描述: 就业时间较短，收入稳定性有待验证
   - 风险权重: 中

## 有利因素分析

- 有固定住所，居住稳定性较好
- 年龄适中，处于职业发展期

## 综合风险评分
风险评分: 75分

依据: 账户透支(-30分) + 高额贷款(-20分) + 就业不稳定(-15分) + 有利因素(+10分)

## 风险管理建议

审批建议: 拒绝

授信额度: 不适用

监控指标: 
- 账户余额变动
- 还款记录

风险缓释措施:
- 建议申请人先改善财务状况
- 可考虑提供担保人
"""
    
    parser = OutputParser()
    parsed = parser.parse(sample_response)
    
    print("\n解析结果:")
    print(f"  风险等级: {parsed.risk_level_text}")
    print(f"  风险评分: {parsed.risk_score}")
    print(f"  审批建议: {parsed.approval_suggestion_text}")
    
    print(f"\n  风险因素 ({len(parsed.risk_factors)}个):")
    for factor in parsed.risk_factors:
        print(f"    - {factor.name}: 权重[{factor.weight}]")
    
    print(f"\n  有利因素 ({len(parsed.favorable_factors)}个):")
    for factor in parsed.favorable_factors:
        print(f"    - {factor.name}")
    
    print(f"\n  监控指标: {parsed.monitoring_indicators}")
    print(f"  缓释措施: {parsed.risk_mitigation_measures}")
    
    print(f"\n  解析成功: {parsed.parse_success}")
    if parsed.parse_errors:
        print(f"  解析错误: {parsed.parse_errors}")
    
    is_valid, errors = parser.validate_result(parsed)
    print(f"\n  结果验证: {'通过' if is_valid else '未通过'}")
    if errors:
        print(f"  验证错误: {errors}")
    
    return parsed


def example_system_statistics():
    """系统统计示例"""
    print("\n" + "=" * 60)
    print("示例6: 系统统计信息")
    print("=" * 60)
    
    rag_system = create_rag_system()
    
    test_data = [
        {"checking_status": "A11", "duration": 24, "credit_amount": 5000, "age": 35},
        {"checking_status": "A14", "duration": 12, "credit_amount": 3000, "age": 45},
        {"checking_status": "A12", "duration": 36, "credit_amount": 8000, "age": 28},
    ]
    
    for i, data in enumerate(test_data):
        rag_system.analyze(data, f"STAT_{i}")
    
    stats = rag_system.get_stats()
    
    print("\n系统统计:")
    print(f"  总分析次数: {stats['total_analyses']}")
    print(f"  总Token消耗: {stats['total_tokens']}")
    print(f"  总费用: ¥{stats['total_cost']:.4f}")
    print(f"  平均耗时: {stats['average_latency']:.2f}秒")
    
    print("\n配置信息:")
    config = stats['config']
    print(f"  检索策略: {config['retrieval_strategy']}")
    print(f"  Top-K: {config['top_k']}")
    print(f"  模板类型: {config['template_type']}")
    
    collection_stats = rag_system.get_collection_stats()
    print("\n知识库统计:")
    for name, info in collection_stats.get('collections', {}).items():
        print(f"  {name}: {info.get('count', 0)} 条记录")
    
    return stats


def run_all_examples():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("RAG风险分析系统使用示例")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    examples = [
        ("基本使用", example_basic_usage),
        ("自定义配置", example_custom_config),
        ("批量预测", example_batch_prediction),
        ("多案例分析", example_multiple_cases),
        ("输出解析", example_output_parsing),
        ("系统统计", example_system_statistics),
    ]
    
    results = {}
    
    for name, func in examples:
        try:
            print(f"\n>>> 运行示例: {name}")
            result = func()
            results[name] = {"success": True, "result": result}
        except Exception as e:
            print(f"\n示例 '{name}' 运行失败: {e}")
            results[name] = {"success": False, "error": str(e)}
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r["success"])
    print(f"\n成功: {success_count}/{len(examples)}")
    
    return results


if __name__ == "__main__":
    results = run_all_examples()
