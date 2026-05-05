import sys
import os
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.volcengine_api import VolcEngineAPI, TokenUsage, CostEstimate
from models.rag.llm_only import LLMRiskAnalyzer, RiskAnalysisResult, BatchAnalysisSummary, print_summary
from config.prompts import RISK_ANALYSIS_SYSTEM_PROMPT, get_risk_analysis_prompt


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_token_stats(usage: TokenUsage, cost: CostEstimate):
    print(f"  Prompt Tokens: {usage.prompt_tokens:,}")
    print(f"  Completion Tokens: {usage.completion_tokens:,}")
    print(f"  Total Tokens: {usage.total_tokens:,}")
    print(f"  费用: ¥{cost.total_cost:.6f} {cost.currency}")


def test_api_connection():
    print_header("测试1: API连接测试")
    
    try:
        api = VolcEngineAPI()
        success, message = api.test_connection()
        
        if success:
            print(f"✓ {message}")
            return True
        else:
            print(f"✗ {message}")
            return False
    except ValueError as e:
        print(f"✗ 配置错误: {e}")
        print("\n请确保以下环境变量已设置:")
        print("  - VOLCENGINE_ACCESS_KEY")
        print("  - VOLCENGINE_SECRET_KEY")
        print("  - VOLCENGINE_ENDPOINT_ID")
        return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False


def test_embedding_api():
    print_header("测试2: Embedding API测试")
    
    try:
        api = VolcEngineAPI()
        
        test_text = "这是一个测试文本，用于验证Embedding API是否正常工作。"
        print(f"测试文本: {test_text}")
        
        start_time = time.time()
        embedding, usage, cost = api.call_embedding(test_text)
        latency = time.time() - start_time
        
        if embedding:
            print(f"✓ Embedding维度: {len(embedding)}")
            print(f"✓ 延迟: {latency:.2f}秒")
            print_token_stats(usage, cost)
            return True
        else:
            print("✗ Embedding为空")
            return False
    except Exception as e:
        print(f"✗ Embedding API测试失败: {e}")
        return False


def test_llm_single_inference():
    print_header("测试3: LLM单条推理测试")
    
    sample_customer = {
        "checking_status": "A11",
        "duration": 24,
        "credit_history": "A32",
        "purpose": "A40",
        "credit_amount": 15000,
        "savings_status": "A64",
        "employment": "A75",
        "installment_commitment": 2,
        "personal_status": "A93",
        "other_parties": "A101",
        "residence_since": 3,
        "property_magnitude": "A121",
        "age": 42,
        "other_payment_plans": "A143",
        "housing": "A152",
        "existing_credits": 2,
        "job": "A174",
        "num_dependents": 2,
        "own_telephone": "A192",
        "foreign_worker": "A202"
    }
    
    print("客户信息:")
    print("-" * 50)
    for key, value in sample_customer.items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    try:
        analyzer = LLMRiskAnalyzer(temperature=0.3)
        
        print("\n正在调用LLM进行分析...")
        start_time = time.time()
        result = analyzer.analyze_single(sample_customer, "TEST_SINGLE_001")
        total_time = time.time() - start_time
        
        print(f"\n✓ 分析完成 (总耗时: {total_time:.2f}秒)")
        print("-" * 50)
        print(f"  申请ID: {result.application_id}")
        print(f"  风险等级: {result.risk_level_text}")
        print(f"  风险评分: {result.risk_score}/100")
        print(f"  审批建议: {result.approval_suggestion}")
        print(f"  延迟: {result.latency:.2f}秒")
        print(f"  Token消耗: {result.total_tokens:,}")
        print(f"  费用: ¥{result.cost:.6f}")
        
        print("\n原始响应预览 (前500字符):")
        print("-" * 50)
        print(result.raw_response[:500] + "..." if len(result.raw_response) > 500 else result.raw_response)
        
        return True
    except Exception as e:
        print(f"✗ LLM单条推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference():
    print_header("测试4: 批量推理测试")
    
    batch_customers = [
        {
            "checking_status": "A14",
            "duration": 6,
            "credit_history": "A31",
            "purpose": "A41",
            "credit_amount": 3000,
            "savings_status": "A64",
            "employment": "A75",
            "installment_commitment": 1,
            "personal_status": "A94",
            "other_parties": "A101",
            "residence_since": 4,
            "property_magnitude": "A121",
            "age": 55,
            "other_payment_plans": "A143",
            "housing": "A152",
            "existing_credits": 1,
            "job": "A174",
            "num_dependents": 2,
            "own_telephone": "A192",
            "foreign_worker": "A202"
        },
        {
            "checking_status": "A11",
            "duration": 48,
            "credit_history": "A34",
            "purpose": "A46",
            "credit_amount": 12000,
            "savings_status": "A61",
            "employment": "A72",
            "installment_commitment": 4,
            "personal_status": "A93",
            "other_parties": "A101",
            "residence_since": 1,
            "property_magnitude": "A124",
            "age": 25,
            "other_payment_plans": "A141",
            "housing": "A151",
            "existing_credits": 3,
            "job": "A171",
            "num_dependents": 1,
            "own_telephone": "A191",
            "foreign_worker": "A201"
        },
        {
            "checking_status": "A12",
            "duration": 18,
            "credit_history": "A32",
            "purpose": "A42",
            "credit_amount": 6000,
            "savings_status": "A62",
            "employment": "A73",
            "installment_commitment": 2,
            "personal_status": "A92",
            "other_parties": "A103",
            "residence_since": 2,
            "property_magnitude": "A122",
            "age": 38,
            "other_payment_plans": "A143",
            "housing": "A152",
            "existing_credits": 1,
            "job": "A173",
            "num_dependents": 1,
            "own_telephone": "A192",
            "foreign_worker": "A202"
        }
    ]
    
    application_ids = ["BATCH_001", "BATCH_002", "BATCH_003"]
    
    print(f"批量测试样本数: {len(batch_customers)}")
    
    try:
        analyzer = LLMRiskAnalyzer(temperature=0.3)
        
        print("\n开始批量分析...")
        start_time = time.time()
        results, summary = analyzer.analyze_batch(
            batch_customers,
            application_ids=application_ids,
            show_progress=True,
            delay_between_calls=0.5
        )
        total_time = time.time() - start_time
        
        print(f"\n✓ 批量分析完成 (总耗时: {total_time:.2f}秒)")
        
        print_summary(summary)
        
        print("\n各样本分析结果:")
        print("-" * 70)
        for r in results:
            print(f"  [{r.application_id}] 风险等级: {r.risk_level_text}, "
                  f"评分: {r.risk_score}, 建议: {r.approval_suggestion}")
        
        return True
    except Exception as e:
        print(f"✗ 批量推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_token_counting():
    print_header("测试5: Token计数功能测试")
    
    try:
        api = VolcEngineAPI()
        
        test_texts = [
            "这是一个简短的测试文本。",
            "这是一个较长的测试文本，包含更多的内容，用于测试Token计数功能是否正常工作。" * 5,
            "This is an English test text for token counting functionality verification.",
        ]
        
        print("Token计数测试:")
        print("-" * 50)
        for i, text in enumerate(test_texts, 1):
            token_count = api.count_tokens(text)
            char_count = len(text)
            print(f"  文本{i}: 字符数={char_count}, Token数={token_count}")
        
        print("\n✓ Token计数功能正常")
        return True
    except Exception as e:
        print(f"✗ Token计数测试失败: {e}")
        return False


def test_cost_estimation():
    print_header("测试6: 费用估算功能测试")
    
    try:
        api = VolcEngineAPI()
        
        test_cases = [
            {"input": 1000, "output": 500},
            {"input": 5000, "output": 2000},
            {"input": 10000, "output": 5000},
        ]
        
        print("LLM费用估算 (doubao-seed-2-0-pro-260215):")
        print("-" * 50)
        for case in test_cases:
            cost = api.estimate_cost(api.llm_model, case["input"], case["output"])
            print(f"  输入{case['input']} tokens + 输出{case['output']} tokens = ¥{cost.total_cost:.6f}")
        
        print("\nEmbedding费用估算 (doubao-embedding-vision-251215):")
        print("-" * 50)
        for case in test_cases:
            cost = api.estimate_cost(api.embedding_model, case["input"], 0)
            print(f"  输入{case['input']} tokens = ¥{cost.total_cost:.6f}")
        
        print("\n✓ 费用估算功能正常")
        return True
    except Exception as e:
        print(f"✗ 费用估算测试失败: {e}")
        return False


def test_api_stats():
    print_header("测试7: API统计功能测试")
    
    try:
        api = VolcEngineAPI()
        
        api.reset_stats()
        
        print("执行一系列API调用以测试统计功能...")
        
        embedding, _, _ = api.call_embedding("测试文本1")
        embedding, _, _ = api.call_embedding("测试文本2")
        
        response, _, _ = api.call_llm("测试问题1", max_tokens=100)
        
        stats = api.get_stats()
        
        print("\nAPI调用统计:")
        print("-" * 50)
        print(f"  总调用次数: {stats['total_calls']}")
        print(f"  成功次数: {stats['successful_calls']}")
        print(f"  失败次数: {stats['failed_calls']}")
        print(f"  成功率: {stats['success_rate']:.1f}%")
        print(f"  总Token消耗: {stats['total_tokens']['total_tokens']:,}")
        print(f"  总费用: ¥{stats['total_cost']['total_cost']:.6f}")
        print(f"  平均延迟: {stats['average_latency']:.2f}秒")
        
        print("\n✓ API统计功能正常")
        return True
    except Exception as e:
        print(f"✗ API统计测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    print("\n" + "=" * 70)
    print("  火山引擎API测试套件")
    print("  模型: doubao-seed-2-0-pro-260215 (LLM)")
    print("        doubao-embedding-vision-251215 (Embedding)")
    print("=" * 70)
    
    results = {}
    
    results["API连接"] = test_api_connection()
    
    if results["API连接"]:
        results["Embedding API"] = test_embedding_api()
        results["LLM单条推理"] = test_llm_single_inference()
        results["批量推理"] = test_batch_inference()
        results["Token计数"] = test_token_counting()
        results["费用估算"] = test_cost_estimation()
        results["API统计"] = test_api_stats()
    else:
        print("\n跳过其他测试（API连接失败）")
        for test_name in ["Embedding API", "LLM单条推理", "批量推理", "Token计数", "费用估算", "API统计"]:
            results[test_name] = False
    
    print_header("测试结果汇总")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ 通过" if passed_flag else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print("-" * 70)
    print(f"  总计: {passed}/{total} 测试通过")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
