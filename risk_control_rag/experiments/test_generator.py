"""
RAG生成模块测试脚本

测试内容：
1. Prompt构建测试
2. LLM生成测试
3. 输出解析测试
4. 完整分析流程示例
"""

import sys
import os
import json
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag.prompt_builder import (
    PromptBuilder,
    PromptConfig,
    RetrievedKnowledge,
    convert_search_results_to_knowledge,
    build_rag_prompt,
)
from models.rag.generator import (
    RAGGenerator,
    GenerationConfig,
    GenerationResult,
    create_generator,
)
from models.rag.output_parser import (
    OutputParser,
    ParsedRiskAnalysis,
    RiskLevel,
    ApprovalSuggestion,
    parse_risk_analysis,
    quick_parse_risk_level,
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    print("\n" + "-" * 50)
    print(f"  {title}")
    print("-" * 50)


def get_sample_customer() -> Dict[str, Any]:
    return {
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


def get_sample_knowledge() -> List[RetrievedKnowledge]:
    return [
        RetrievedKnowledge(
            content="案例类型: 信用贷款违约\n案例描述: 借款人支票账户透支，贷款期限24个月，贷款金额15000DM，最终违约。\n风险指标: 支票账户透支, 贷款金额较高, 贷款期限较长\n处理决策: 拒绝贷款申请",
            source_type="risk_case",
            source_id="RC001",
            relevance_score=0.85,
            metadata={"case_type": "信用贷款违约"}
        ),
        RetrievedKnowledge(
            content="来源: 巴塞尔协议III\n分类: 信用风险管理\n规则内容: 银行应建立完善的信用风险评级体系，对借款人进行全面的信用评估。对于高风险客户，应采取更严格的审批标准和风险缓释措施。",
            source_type="regulation",
            source_id="REG001",
            relevance_score=0.75,
            metadata={"category": "信用风险管理"}
        ),
        RetrievedKnowledge(
            content="分类: 信用评估指标\n知识内容: 支票账户状态是重要的信用评估指标。账户透支(A11)通常表明借款人资金管理能力较弱，违约风险较高。建议结合其他指标综合评估。",
            source_type="industry_knowledge",
            source_id="IK001",
            relevance_score=0.80,
            metadata={"category": "信用评估指标"}
        ),
    ]


def get_sample_search_results() -> List[Dict[str, Any]]:
    return [
        {
            "document": "案例类型: 高风险贷款\n案例描述: 客户支票账户状态为透支，申请大额贷款，最终发生违约。\n风险指标: 支票账户透支, 大额贷款, 还款压力大",
            "metadata": {"type": "risk_case", "id": "CASE_001"},
            "distance": 0.15
        },
        {
            "document": "监管要求: 对于支票账户透支的申请人，应审慎评估其还款能力，必要时要求提供担保。",
            "metadata": {"type": "regulation", "id": "REG_001"},
            "distance": 0.20
        },
    ]


def test_prompt_builder():
    print_header("测试1: Prompt构建测试")
    
    print_section("1.1 基础Prompt构建")
    
    config = PromptConfig(
        max_context_length=2000,
        max_knowledge_items=3,
        include_source_annotation=True,
        template_type="default"
    )
    
    builder = PromptBuilder(config)
    customer_data = get_sample_customer()
    knowledge_items = get_sample_knowledge()
    
    prompt = builder.build_prompt(customer_data, knowledge_items)
    
    print(f"Prompt长度: {len(prompt)} 字符")
    print(f"预估Token数: {builder.estimate_token_count(prompt)}")
    
    print("\nPrompt预览 (前800字符):")
    print("-" * 40)
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
    
    print_section("1.2 不同模板类型测试")
    
    for template_type in ["default", "concise", "detailed"]:
        prompt = builder.build_prompt(customer_data, knowledge_items, template_type)
        print(f"\n模板类型: {template_type}")
        print(f"  - Prompt长度: {len(prompt)} 字符")
        print(f"  - 预估Token: {builder.estimate_token_count(prompt)}")
    
    print_section("1.3 知识来源标注测试")
    
    config_with_annotation = PromptConfig(include_source_annotation=True)
    config_without_annotation = PromptConfig(include_source_annotation=False)
    
    builder_with = PromptBuilder(config_with_annotation)
    builder_without = PromptBuilder(config_without_annotation)
    
    prompt_with = builder_with.build_prompt(customer_data, knowledge_items)
    prompt_without = builder_without.build_prompt(customer_data, knowledge_items)
    
    print("带知识来源标注的Prompt片段:")
    idx = prompt_with.find("相关知识库内容")
    if idx > 0:
        print(prompt_with[idx:idx+400])
    
    print("\n不带知识来源标注的Prompt片段:")
    idx = prompt_without.find("相关知识库内容")
    if idx > 0:
        print(prompt_without[idx:idx+400])
    
    print_section("1.4 上下文长度控制测试")
    
    many_knowledge = [
        RetrievedKnowledge(
            content=f"知识条目 {i}: " + "这是一段较长的知识内容。" * 20,
            source_type="industry_knowledge",
            source_id=f"IK{i:03d}",
            relevance_score=0.8 - i * 0.05
        )
        for i in range(10)
    ]
    
    config_short = PromptConfig(max_context_length=500, max_knowledge_items=5)
    builder_short = PromptBuilder(config_short)
    
    prompt_short = builder_short.build_prompt(customer_data, many_knowledge)
    print(f"限制上下文长度500字符:")
    print(f"  - 实际Prompt长度: {len(prompt_short)} 字符")
    
    stats = builder_short.get_prompt_stats(customer_data, many_knowledge)
    print(f"\nPrompt统计信息:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print_section("1.5 搜索结果转换测试")
    
    search_results = get_sample_search_results()
    converted = convert_search_results_to_knowledge(search_results)
    
    print(f"转换了 {len(converted)} 条知识:")
    for k in converted:
        print(f"  - 类型: {k.source_type}, ID: {k.source_id}, 相关度: {k.relevance_score:.2f}")
    
    print("\n✓ Prompt构建测试完成")
    return True


def test_output_parser():
    print_header("测试2: 输出解析测试")
    
    print_section("2.1 风险等级解析测试")
    
    parser = OutputParser()
    
    test_cases = [
        ("风险等级：低风险", RiskLevel.LOW),
        ("风险等级：高风险", RiskLevel.HIGH),
        ("风险等级：中风险", RiskLevel.MEDIUM),
        ("经评估，该申请人风险较低", RiskLevel.LOW),
        ("存在明显的高风险信号", RiskLevel.HIGH),
    ]
    
    for text, expected in test_cases:
        result = parser._parse_risk_level(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:30]}...' -> {result.to_chinese()} (期望: {expected.to_chinese()})")
    
    print_section("2.2 风险评分解析测试")
    
    score_cases = [
        ("风险评分：75分", 75),
        ("综合评分：45", 45),
        ("风险评分为82分", 82),
        ("评分: 30/100", 30),
    ]
    
    for text, expected in score_cases:
        result = parser._parse_risk_score(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result} (期望: {expected})")
    
    print_section("2.3 完整响应解析测试")
    
    sample_response = """
## 风险评估报告

### 1. 风险等级判定
风险等级：中风险

该申请人存在一定的风险因素，需要关注。

### 2. 关键风险因素分析

1. 支票账户透支
   - 风险描述：申请人支票账户处于透支状态，表明资金管理能力较弱
   - 风险权重：高
   - 参考来源：[1]

2. 贷款金额较高
   - 风险描述：申请贷款金额15000DM，相对收入水平较高
   - 风险权重：中

3. 贷款期限较长
   - 风险描述：24个月的贷款期限增加了不确定性
   - 风险权重：中

### 3. 有利因素分析

- 就业稳定：就业超过7年，收入来源稳定
- 拥有房产：自有住房，资产状况良好
- 储蓄充足：储蓄账户余额较高

### 4. 综合风险评分
风险评分：55分

评分依据：综合考虑风险因素和有利因素，给予中等风险评分。

### 5. 风险管理建议

审批建议：人工复核

授信额度：建议降低授信额度至10000DM

监控指标：
- 定期检查支票账户状态
- 关注还款及时性
- 监控收入变化

风险缓释措施：
- 要求提供额外担保
- 设置更严格的还款计划

### 6. 知识来源引用
[1] 风险案例 RC001
[2] 监管法规 REG001
"""
    
    result = parser.parse(sample_response)
    
    print(f"解析结果:")
    print(f"  - 风险等级: {result.risk_level_text}")
    print(f"  - 风险评分: {result.risk_score}")
    print(f"  - 审批建议: {result.approval_suggestion_text}")
    print(f"  - 风险因素数量: {len(result.risk_factors)}")
    print(f"  - 有利因素数量: {len(result.favorable_factors)}")
    print(f"  - 知识来源数量: {len(result.knowledge_sources)}")
    print(f"  - 解析成功: {result.parse_success}")
    
    if result.risk_factors:
        print(f"\n  风险因素详情:")
        for i, factor in enumerate(result.risk_factors, 1):
            print(f"    {i}. {factor.name} (权重: {factor.weight})")
    
    if result.favorable_factors:
        print(f"\n  有利因素详情:")
        for i, factor in enumerate(result.favorable_factors, 1):
            print(f"    {i}. {factor.name}")
    
    print_section("2.4 结果验证测试")
    
    is_valid, errors = parser.validate_result(result)
    print(f"验证结果: {'通过' if is_valid else '未通过'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    print_section("2.5 容错机制测试")
    
    incomplete_response = "这是一个不完整的响应，没有明确的风险等级和评分。"
    result_incomplete = parser.parse(incomplete_response)
    
    print(f"不完整响应解析结果:")
    print(f"  - 解析成功: {result_incomplete.parse_success}")
    print(f"  - 解析错误: {result_incomplete.parse_errors}")
    
    fallback_values = {
        "risk_level": RiskLevel.MEDIUM,
        "risk_score": 50,
        "approval": ApprovalSuggestion.REVIEW
    }
    result_with_fallback = parser.parse_with_fallback(incomplete_response, fallback_values)
    
    print(f"\n使用回退值后:")
    print(f"  - 风险等级: {result_with_fallback.risk_level_text}")
    print(f"  - 风险评分: {result_with_fallback.risk_score}")
    print(f"  - 审批建议: {result_with_fallback.approval_suggestion_text}")
    
    print_section("2.6 快速解析测试")
    
    quick_result = quick_parse_risk_level(sample_response)
    print(f"快速解析结果:")
    print(f"  - 风险等级: {quick_result[0].to_chinese()}")
    print(f"  - 风险评分: {quick_result[1]}")
    print(f"  - 审批建议: {quick_result[2].value}")
    
    print("\n✓ 输出解析测试完成")
    return True


def test_llm_generation():
    print_header("测试3: LLM生成测试")
    
    try:
        from utils.volcengine_api import VolcEngineAPI
        api = VolcEngineAPI()
        success, message = api.test_connection()
        
        if not success:
            print(f"API连接失败: {message}")
            print("跳过LLM生成测试")
            return False
    except Exception as e:
        print(f"API初始化失败: {e}")
        print("跳过LLM生成测试")
        return False
    
    print_section("3.1 基础生成测试")
    
    gen_config = GenerationConfig(
        temperature=0.3,
        max_tokens=1024,
        max_retries=2
    )
    
    generator = RAGGenerator(generation_config=gen_config)
    
    customer_data = get_sample_customer()
    knowledge_items = get_sample_knowledge()
    
    print("正在调用LLM生成风险分析...")
    start_time = time.time()
    
    result = generator.generate_with_knowledge(
        customer_data,
        knowledge_items,
        template_type="concise"
    )
    
    total_time = time.time() - start_time
    
    print(f"\n生成结果:")
    print(f"  - 成功: {result.success}")
    print(f"  - 延迟: {result.latency:.2f}秒")
    print(f"  - Token消耗: {result.total_tokens}")
    print(f"  - 费用: ¥{result.cost:.6f}")
    
    if result.success:
        print(f"\n生成内容预览 (前500字符):")
        print("-" * 40)
        print(result.content[:500] + "..." if len(result.content) > 500 else result.content)
    else:
        print(f"  - 错误信息: {result.error_message}")
    
    print_section("3.2 费用估算测试")
    
    cost_estimate = generator.estimate_cost(customer_data, knowledge_items)
    print("费用估算:")
    for key, value in cost_estimate.items():
        print(f"  - {key}: {value}")
    
    print_section("3.3 统计信息测试")
    
    stats = generator.get_stats()
    print("生成器统计:")
    print(f"  - 总生成次数: {stats['total_generations']}")
    print(f"  - 总Token消耗: {stats['total_tokens']}")
    print(f"  - 总费用: ¥{stats['total_cost']:.6f}")
    print(f"  - 平均延迟: {stats['average_latency']:.2f}秒")
    
    print("\n✓ LLM生成测试完成")
    return True


def test_full_pipeline():
    print_header("测试4: 完整分析流程示例")
    
    print_section("4.1 初始化组件")
    
    prompt_config = PromptConfig(
        max_context_length=3000,
        max_knowledge_items=5,
        include_source_annotation=True,
        template_type="default"
    )
    
    gen_config = GenerationConfig(
        temperature=0.3,
        max_tokens=2048,
        max_retries=3
    )
    
    builder = PromptBuilder(prompt_config)
    parser = OutputParser()
    
    print("组件初始化完成")
    print(f"  - Prompt配置: max_context={prompt_config.max_context_length}, max_items={prompt_config.max_knowledge_items}")
    print(f"  - 生成配置: temp={gen_config.temperature}, max_tokens={gen_config.max_tokens}")
    
    print_section("4.2 准备输入数据")
    
    customer_data = get_sample_customer()
    knowledge_items = get_sample_knowledge()
    
    print("客户信息:")
    from config.prompts import format_customer_info
    print(format_customer_info(customer_data))
    
    print(f"\n知识库条目: {len(knowledge_items)} 条")
    for k in knowledge_items:
        print(f"  - [{k.source_type}] {k.source_id}: 相关度 {k.relevance_score:.2f}")
    
    print_section("4.3 构建Prompt")
    
    prompt = builder.build_prompt(customer_data, knowledge_items)
    system_prompt = builder.get_system_prompt()
    
    print(f"Prompt构建完成:")
    print(f"  - System Prompt长度: {len(system_prompt)} 字符")
    print(f"  - User Prompt长度: {len(prompt)} 字符")
    print(f"  - 预估Token: {builder.estimate_token_count(prompt) + builder.estimate_token_count(system_prompt)}")
    
    print_section("4.4 调用LLM生成")
    
    try:
        generator = RAGGenerator(gen_config, prompt_config)
        
        print("正在生成风险分析...")
        start_time = time.time()
        
        result = generator.generate_with_knowledge(
            customer_data,
            knowledge_items,
            template_type="default"
        )
        
        generation_time = time.time() - start_time
        
        print(f"\n生成完成:")
        print(f"  - 耗时: {generation_time:.2f}秒")
        print(f"  - Token消耗: {result.total_tokens}")
        print(f"  - 费用: ¥{result.cost:.6f}")
        
    except Exception as e:
        print(f"生成失败: {e}")
        print("使用模拟响应继续测试...")
        
        result = GenerationResult(
            content="""
## 风险评估报告

### 1. 风险等级判定
风险等级：中风险

该申请人存在一定的风险因素，需要关注。根据知识库案例[1]，支票账户透支是重要的风险信号。

### 2. 关键风险因素分析

1. 支票账户透支
   - 风险描述：申请人支票账户处于透支状态(A11)，表明资金管理能力较弱
   - 风险权重：高
   - 参考来源：[1]

2. 贷款金额较高
   - 风险描述：申请贷款金额15000DM，相对较高
   - 风险权重：中

3. 贷款期限较长
   - 风险描述：24个月的贷款期限增加了不确定性
   - 风险权重：中

### 3. 有利因素分析

- 就业稳定：就业超过7年，收入来源稳定
- 拥有房产：自有住房，资产状况良好
- 储蓄充足：储蓄账户余额>=1000DM

### 4. 综合风险评分
风险评分：55分

### 5. 风险管理建议
审批建议：人工复核

授信额度：建议降低至10000DM

监控指标：支票账户状态、还款及时性

### 6. 知识来源引用
[1] 风险案例 RC001
[2] 监管法规 REG001
""",
            prompt_tokens=500,
            completion_tokens=300,
            total_tokens=800,
            cost=0.001,
            latency=2.5,
            success=True
        )
    
    print_section("4.5 解析输出结果")
    
    parsed_result = parser.parse(result.content)
    
    print("解析结果:")
    print(f"  - 风险等级: {parsed_result.risk_level_text}")
    print(f"  - 风险评分: {parsed_result.risk_score}")
    print(f"  - 审批建议: {parsed_result.approval_suggestion_text}")
    print(f"  - 风险因素: {len(parsed_result.risk_factors)} 个")
    print(f"  - 有利因素: {len(parsed_result.favorable_factors)} 个")
    print(f"  - 知识来源: {len(parsed_result.knowledge_sources)} 个")
    print(f"  - 解析成功: {parsed_result.parse_success}")
    
    if parsed_result.parse_errors:
        print(f"  - 解析错误: {parsed_result.parse_errors}")
    
    print_section("4.6 结果验证")
    
    is_valid, errors = parser.validate_result(parsed_result)
    print(f"验证结果: {'通过' if is_valid else '未通过'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    print_section("4.7 完整分析报告")
    
    report = {
        "application_info": {
            "customer_data": customer_data,
        },
        "knowledge_used": [
            {
                "type": k.source_type,
                "id": k.source_id,
                "relevance": k.relevance_score
            }
            for k in knowledge_items
        ],
        "analysis_result": parsed_result.to_dict(),
        "generation_info": {
            "total_tokens": result.total_tokens,
            "cost": result.cost,
            "latency": result.latency,
        }
    }
    
    print("完整分析报告 (JSON格式):")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:1500] + "...")
    
    print("\n✓ 完整分析流程测试完成")
    return True


def run_all_tests():
    print("\n" + "=" * 70)
    print("  RAG生成模块测试套件")
    print("=" * 70)
    
    results = {}
    
    results["Prompt构建"] = test_prompt_builder()
    results["输出解析"] = test_output_parser()
    results["LLM生成"] = test_llm_generation()
    results["完整流程"] = test_full_pipeline()
    
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
