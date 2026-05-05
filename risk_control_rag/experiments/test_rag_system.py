"""
RAG系统测试脚本

测试完整的RAG风险分析流程，包括：
- 单条分析测试
- 批量预测测试
- 不同检索策略对比
- Token消耗和费用记录
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag.rag_system import RAGRiskSystem, RAGSystemConfig, RetrievalStrategy, create_rag_system
from models.rag.batch_predictor import BatchPredictor, BatchPredictorConfig, create_batch_predictor


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'-' * 60}")


def print_result_summary(result: dict):
    """打印结果摘要"""
    print(f"\n风险等级: {result.get('risk_level', '未知')}")
    print(f"风险评分: {result.get('risk_score', -1)}")
    print(f"审批建议: {result.get('approval_suggestion', '待定')}")
    
    risk_factors = result.get('risk_factors', [])
    if risk_factors:
        print(f"\n主要风险因素 ({len(risk_factors)}个):")
        for i, factor in enumerate(risk_factors[:3], 1):
            print(f"  {i}. {factor.get('name', '未知')}: {factor.get('description', '')[:50]}...")
    
    token_usage = result.get('token_usage', {})
    if token_usage:
        print(f"\nToken消耗: 输入{token_usage.get('prompt_tokens', 0)} + 输出{token_usage.get('completion_tokens', 0)} = {token_usage.get('total_tokens', 0)}")
    
    cost = result.get('cost', 0)
    if cost:
        print(f"费用估算: ¥{cost:.4f}")
    
    print(f"耗时: {result.get('latency', 0):.2f}秒")


def get_test_customers() -> list:
    """获取测试客户数据"""
    return [
        {
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
        },
        {
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
        },
        {
            "checking_status": "A12",
            "duration": 36,
            "credit_history": "A33",
            "purpose": "A41",
            "credit_amount": 12000,
            "savings_status": "A61",
            "employment": "A72",
            "installment_commitment": 4,
            "personal_status": "A93",
            "other_parties": "A101",
            "residence_since": 1,
            "property_magnitude": "A123",
            "age": 28,
            "other_payment_plans": "A141",
            "housing": "A151",
            "existing_credits": 1,
            "job": "A172",
            "num_dependents": 0,
            "own_telephone": "A191",
            "foreign_worker": "A201",
        },
    ]


def test_single_analysis(rag_system: RAGRiskSystem):
    """测试单条分析"""
    print_separator("测试1: 单条风险分析")
    
    test_customers = get_test_customers()
    customer = test_customers[0]
    
    print("\n客户信息:")
    print(f"  支票账户状态: 账户透支")
    print(f"  贷款期限: 24个月")
    print(f"  信用历史: 现有贷款按时还款")
    print(f"  贷款目的: 购买收音机/电视")
    print(f"  贷款金额: 5000 DM")
    print(f"  就业状态: 就业1-4年")
    print(f"  年龄: 35岁")
    
    print("\n正在分析中...")
    start_time = time.time()
    
    result = rag_system.analyze(
        customer_data=customer,
        customer_id="TEST_001",
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n分析完成，总耗时: {elapsed:.2f}秒")
    
    result_dict = result.to_dict()
    print_result_summary(result_dict)
    
    print("\n原始响应片段:")
    raw = result_dict.get('raw_response', '')
    print(raw[:500] + "..." if len(raw) > 500 else raw)
    
    return result


def test_batch_prediction(rag_system: RAGRiskSystem):
    """测试批量预测"""
    print_separator("测试2: 批量预测")
    
    test_customers = get_test_customers()
    customer_ids = [f"TEST_{i:03d}" for i in range(len(test_customers))]
    
    print(f"\n批量预测 {len(test_customers)} 条数据...")
    
    def progress_callback(current: int, total: int, message: str):
        pct = current / total * 100
        print(f"\r  进度: {current}/{total} ({pct:.0f}%) - {message}", end="", flush=True)
    
    predictor = create_batch_predictor(
        rag_system=rag_system,
        cache_enabled=True,
        checkpoint_enabled=True,
        progress_callback=progress_callback,
    )
    
    start_time = time.time()
    
    batch_result = predictor.predict_batch(
        batch_data=test_customers,
        customer_ids=customer_ids,
        batch_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    elapsed = time.time() - start_time
    print(f"\n\n批量预测完成，总耗时: {elapsed:.2f}秒")
    
    print("\n批量预测结果摘要:")
    print(f"  总数: {batch_result.total_count}")
    print(f"  成功: {batch_result.success_count}")
    print(f"  失败: {batch_result.failed_count}")
    print(f"  缓存命中: {batch_result.cached_count}")
    print(f"  总Token: {batch_result.total_tokens}")
    print(f"  总费用: ¥{batch_result.total_cost:.4f}")
    
    print("\n各样本结果:")
    for i, result in enumerate(batch_result.results):
        print(f"\n  [{i+1}] 客户ID: {result.get('customer_id', 'N/A')}")
        print(f"      风险等级: {result.get('risk_level', '未知')}")
        print(f"      风险评分: {result.get('risk_score', -1)}")
        print(f"      审批建议: {result.get('approval_suggestion', '待定')}")
    
    report = predictor.generate_report(batch_result)
    print("\n" + report)
    
    return batch_result


def test_retrieval_strategies():
    """测试不同检索策略"""
    print_separator("测试3: 检索策略对比")
    
    test_customer = get_test_customers()[0]
    
    strategies = [
        ("向量检索", RetrievalStrategy.VECTOR),
        ("混合检索", RetrievalStrategy.HYBRID),
    ]
    
    results = {}
    
    for name, strategy in strategies:
        print(f"\n使用 {name} 策略...")
        
        config = RAGSystemConfig(
            retrieval_strategy=strategy,
            top_k=5,
            temperature=0.3,
        )
        
        rag_system = create_rag_system(
            retrieval_strategy=strategy.value,
            top_k=5,
        )
        
        start_time = time.time()
        result = rag_system.analyze(test_customer, f"STRATEGY_{strategy.value}")
        elapsed = time.time() - start_time
        
        results[name] = {
            "result": result,
            "elapsed": elapsed,
        }
        
        result_dict = result.to_dict()
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  风险等级: {result_dict.get('risk_level', '未知')}")
        print(f"  风险评分: {result_dict.get('risk_score', -1)}")
        print(f"  Token消耗: {result_dict.get('token_usage', {}).get('total_tokens', 0)}")
        print(f"  费用: ¥{result_dict.get('cost', 0):.4f}")
    
    print("\n策略对比总结:")
    print(f"{'策略':<15} {'耗时':<10} {'风险等级':<10} {'风险评分':<10}")
    print("-" * 50)
    for name, data in results.items():
        result_dict = data["result"].to_dict()
        print(f"{name:<15} {data['elapsed']:<10.2f} {result_dict.get('risk_level', '未知'):<10} {result_dict.get('risk_score', -1):<10}")
    
    return results


def test_template_types():
    """测试不同Prompt模板"""
    print_separator("测试4: Prompt模板对比")
    
    test_customer = get_test_customers()[0]
    
    templates = ["default", "concise", "detailed"]
    
    rag_system = create_rag_system()
    
    results = {}
    
    for template in templates:
        print(f"\n使用 {template} 模板...")
        
        rag_system.set_template_type(template)
        
        start_time = time.time()
        result = rag_system.analyze(test_customer, f"TEMPLATE_{template}")
        elapsed = time.time() - start_time
        
        results[template] = {
            "result": result,
            "elapsed": elapsed,
        }
        
        result_dict = result.to_dict()
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  响应长度: {len(result_dict.get('raw_response', ''))} 字符")
        print(f"  Token消耗: {result_dict.get('token_usage', {}).get('total_tokens', 0)}")
    
    print("\n模板对比总结:")
    print(f"{'模板':<15} {'耗时':<10} {'响应长度':<12} {'Token消耗':<12}")
    print("-" * 50)
    for template, data in results.items():
        result_dict = data["result"].to_dict()
        print(f"{template:<15} {data['elapsed']:<10.2f} {len(result_dict.get('raw_response', '')):<12} {result_dict.get('token_usage', {}).get('total_tokens', 0):<12}")
    
    return results


def test_system_stats():
    """测试系统统计"""
    print_separator("测试5: 系统统计信息")
    
    rag_system = create_rag_system()
    
    test_customers = get_test_customers()
    
    for i, customer in enumerate(test_customers):
        rag_system.analyze(customer, f"STAT_TEST_{i}")
    
    stats = rag_system.get_stats()
    
    print("\n系统统计信息:")
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
    print(f"  持久化目录: {collection_stats.get('persist_dir', 'N/A')}")
    collections = collection_stats.get('collections', {})
    for name, info in collections.items():
        print(f"  {name}: {info.get('count', 0)} 条记录")
    
    return stats


def test_without_retrieval():
    """测试无检索模式（纯LLM）"""
    print_separator("测试6: 纯LLM模式（无检索）")
    
    test_customer = get_test_customers()[0]
    
    rag_system = create_rag_system()
    
    print("\n使用RAG模式...")
    start_time = time.time()
    rag_result = rag_system.analyze(test_customer, "RAG_MODE")
    rag_elapsed = time.time() - start_time
    
    print("\n使用纯LLM模式（无检索）...")
    start_time = time.time()
    llm_result = rag_system.analyze_without_retrieval(test_customer, "LLM_ONLY")
    llm_elapsed = time.time() - start_time
    
    print("\n对比结果:")
    rag_dict = rag_result.to_dict()
    llm_dict = llm_result.to_dict()
    
    print(f"\n{'模式':<15} {'耗时':<10} {'风险等级':<10} {'风险评分':<10} {'Token':<10}")
    print("-" * 60)
    print(f"{'RAG':<15} {rag_elapsed:<10.2f} {rag_dict.get('risk_level', '未知'):<10} {rag_dict.get('risk_score', -1):<10} {rag_dict.get('token_usage', {}).get('total_tokens', 0):<10}")
    print(f"{'纯LLM':<15} {llm_elapsed:<10.2f} {llm_dict.get('risk_level', '未知'):<10} {llm_dict.get('risk_score', -1):<10} {llm_dict.get('token_usage', {}).get('total_tokens', 0):<10}")
    
    return {"rag": rag_result, "llm_only": llm_result}


def run_all_tests():
    """运行所有测试"""
    print_separator("RAG风险分析系统测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    try:
        print("\n初始化RAG系统...")
        rag_system = create_rag_system(
            retrieval_strategy="hybrid",
            top_k=5,
            temperature=0.3,
        )
        print("系统初始化完成")
        
        all_results["single_analysis"] = test_single_analysis(rag_system)
        
        all_results["batch_prediction"] = test_batch_prediction(rag_system)
        
        all_results["retrieval_strategies"] = test_retrieval_strategies()
        
        all_results["template_types"] = test_template_types()
        
        all_results["system_stats"] = test_system_stats()
        
        all_results["without_retrieval"] = test_without_retrieval()
        
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    print_separator("测试完成")
    
    return all_results


if __name__ == "__main__":
    results = run_all_tests()
