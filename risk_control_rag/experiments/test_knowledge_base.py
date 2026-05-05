import os
import sys
import json
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.knowledge_base import KnowledgeBase


def print_separator(title: str = ""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def print_search_results(results: List[Dict], query: str):
    print(f"\n查询: {query}")
    print("-" * 60)
    
    if not results:
        print("未找到相关结果")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n【结果 {i}】")
        print(f"相似度距离: {result['distance']:.4f}")
        print(f"元数据: {json.dumps(result['metadata'], ensure_ascii=False, indent=2)}")
        print(f"内容:\n{result['document']}")


def test_risk_cases_search(kb: KnowledgeBase):
    print_separator("测试风险案例检索")
    
    test_queries = [
        "如何识别欺诈贷款申请？",
        "洗钱的主要特征有哪些？",
        "企业信用违约的预警信号",
        "如何处理多头借贷问题？",
        "虚假贸易背景的特征"
    ]
    
    for query in test_queries:
        results = kb.search(query, collection_name="risk_cases", top_k=3)
        print_search_results(results, query)
        print("\n" + "-" * 60)


def test_regulations_search(kb: KnowledgeBase):
    print_separator("测试监管规则检索")
    
    test_queries = [
        "反洗钱法对客户身份识别的要求",
        "商业银行贷款审批流程规定",
        "贷款风险分类标准",
        "大额交易报告标准",
        "关联交易管理规定"
    ]
    
    for query in test_queries:
        results = kb.search(query, collection_name="regulations", top_k=3)
        print_search_results(results, query)
        print("\n" + "-" * 60)


def test_industry_knowledge_search(kb: KnowledgeBase):
    print_separator("测试行业知识检索")
    
    test_queries = [
        "信用风险评估的5C原则是什么？",
        "如何进行财务报表分析？",
        "反洗钱监测的三个阶段",
        "贷后管理的主要内容",
        "担保物价值评估方法"
    ]
    
    for query in test_queries:
        results = kb.search(query, collection_name="industry_knowledge", top_k=3)
        print_search_results(results, query)
        print("\n" + "-" * 60)


def test_cross_collection_search(kb: KnowledgeBase):
    print_separator("测试跨集合检索")
    
    test_queries = [
        "如何防范贷款欺诈风险？",
        "反洗钱可疑交易识别方法",
        "企业信用风险评估要点"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 60)
        
        all_results = kb.search_all_collections(query, top_k=2)
        
        for collection_name, results in all_results.items():
            if results:
                print(f"\n【{collection_name} 集合结果】")
                for i, result in enumerate(results, 1):
                    print(f"  [{i}] 距离: {result['distance']:.4f}")
                    print(f"      ID: {result['metadata'].get('id', 'N/A')}")
                    doc_preview = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
                    print(f"      内容预览: {doc_preview}")


def test_filtered_search(kb: KnowledgeBase):
    print_separator("测试过滤检索")
    
    print("\n1. 按案例类型过滤（欺诈类型）:")
    results = kb.search_by_case_type("贷款申请异常", case_type="欺诈", top_k=3)
    print_search_results(results, "贷款申请异常（案例类型：欺诈）")
    
    print("\n2. 按案例类型过滤（洗钱类型）:")
    results = kb.search_by_case_type("大额资金交易", case_type="洗钱", top_k=3)
    print_search_results(results, "大额资金交易（案例类型：洗钱）")
    
    print("\n3. 按分类过滤（反洗钱类别）:")
    results = kb.search_by_category("客户身份识别", category="反洗钱", collection_name="regulations", top_k=3)
    print_search_results(results, "客户身份识别（分类：反洗钱）")


def test_retrieval_quality(kb: KnowledgeBase):
    print_separator("测试检索质量")
    
    test_cases = [
        {
            "query": "伪造收入证明骗贷",
            "expected_type": "欺诈",
            "expected_keywords": ["伪造", "收入证明", "贷款"]
        },
        {
            "query": "企业账户大额频繁交易洗钱",
            "expected_type": "洗钱",
            "expected_keywords": ["大额", "交易", "洗钱"]
        },
        {
            "query": "借款人连续逾期不还款",
            "expected_type": "信用违约",
            "expected_keywords": ["逾期", "还款"]
        }
    ]
    
    print("\n检索质量评估:")
    print("-" * 60)
    
    for case in test_cases:
        results = kb.search(case["query"], collection_name="risk_cases", top_k=1)
        
        if results:
            result = results[0]
            actual_type = result["metadata"].get("case_type", "")
            document = result["document"]
            
            type_match = actual_type == case["expected_type"]
            
            keyword_matches = sum(1 for kw in case["expected_keywords"] if kw in document)
            keyword_match_rate = keyword_matches / len(case["expected_keywords"])
            
            print(f"\n查询: {case['query']}")
            print(f"  期望类型: {case['expected_type']}")
            print(f"  实际类型: {actual_type}")
            print(f"  类型匹配: {'✓' if type_match else '✗'}")
            print(f"  关键词匹配率: {keyword_match_rate:.0%}")
            print(f"  相似度距离: {result['distance']:.4f}")


def print_knowledge_base_stats(kb: KnowledgeBase):
    print_separator("知识库统计信息")
    
    stats = kb.get_collection_stats()
    
    print("\n各集合文档数量:")
    total = 0
    for name, count in stats.items():
        print(f"  {name}: {count} 条")
        total += count
    print(f"\n  总计: {total} 条")
    
    print("\n数据文件统计:")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base")
    
    files = [
        ("risk_cases.json", "风险案例"),
        ("regulations.json", "监管规则"),
        ("industry_knowledge.json", "行业知识")
    ]
    
    for filename, desc in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  {desc} ({filename}): {len(data)} 条")


def run_all_tests():
    print_separator("知识库检索测试")
    
    kb = KnowledgeBase()
    
    stats = kb.get_collection_stats()
    if all(count == 0 for count in stats.values()):
        print("\n知识库为空，开始构建...")
        kb.build_all_collections(batch_size=10)
    
    print_knowledge_base_stats(kb)
    
    test_risk_cases_search(kb)
    test_regulations_search(kb)
    test_industry_knowledge_search(kb)
    test_cross_collection_search(kb)
    test_filtered_search(kb)
    test_retrieval_quality(kb)
    
    print_separator("测试完成")


if __name__ == "__main__":
    run_all_tests()
