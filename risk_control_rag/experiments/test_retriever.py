"""
检索模块测试脚本

测试内容：
1. 向量检索效果测试
2. 混合检索效果测试
3. 不同检索策略对比
4. 检索示例输出
"""

import os
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag.embedding import DoubaoEmbedding, create_embedding_model, EmbeddingCache
from models.rag.retriever import VectorRetriever, create_retriever, SearchResult
from models.rag.hybrid_retriever import HybridRetriever, create_hybrid_retriever, HybridResult


def print_separator(title: str = ""):
    """打印分隔线"""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)


def test_embedding():
    """测试Embedding功能"""
    print_separator("测试 Embedding 功能")
    
    embedding_model = create_embedding_model(
        cache_enabled=True,
        cache_dir="./cache/embeddings"
    )
    
    test_texts = [
        "这是一条关于欺诈风险的测试文本",
        "洗钱行为通常涉及大额资金转移",
        "信用违约是指借款人无法按时偿还贷款"
    ]
    
    print("\n1. 测试单文本向量化:")
    for text in test_texts:
        start_time = time.time()
        embedding, token_usage, cost = embedding_model.embed_single(text)
        elapsed = time.time() - start_time
        
        if embedding:
            print(f"   文本: '{text[:30]}...'")
            print(f"   向量维度: {len(embedding)}")
            print(f"   耗时: {elapsed:.3f}s")
            print(f"   Token数: {token_usage.total_tokens}")
        else:
            print(f"   向量化失败: '{text[:30]}...'")
        print()
    
    print("\n2. 测试批量向量化:")
    start_time = time.time()
    result = embedding_model.embed_batch(test_texts)
    elapsed = time.time() - start_time
    
    print(f"   文本数量: {len(test_texts)}")
    print(f"   成功向量化: {len([e for e in result.embeddings if e])}")
    print(f"   缓存命中: {result.cached_count}")
    print(f"   API调用次数: {result.api_call_count}")
    print(f"   总耗时: {elapsed:.3f}s")
    print(f"   总Token数: {result.token_usage.total_tokens}")
    
    print("\n3. 缓存统计:")
    cache_stats = embedding_model.get_cache_stats()
    print(f"   内存缓存大小: {cache_stats['memory_cache_size']}")
    print(f"   磁盘缓存大小: {cache_stats['disk_cache_size']}")
    
    print("\n4. API调用统计:")
    api_stats = embedding_model.get_stats()
    print(f"   总调用次数: {api_stats['total_calls']}")
    print(f"   成功率: {api_stats['success_rate']:.2f}%")
    print(f"   总成本: {api_stats['total_cost']['total_cost']:.6f} {api_stats['total_cost']['currency']}")


def test_vector_retriever():
    """测试向量检索功能"""
    print_separator("测试向量检索功能")
    
    retriever = create_retriever(
        top_k=5,
        similarity_threshold=0.3
    )
    
    print("\n1. 向量库统计:")
    stats = retriever.get_stats()
    for name, info in stats.get("collections", {}).items():
        print(f"   {name}: {info['count']} 条文档")
    
    test_queries = [
        "如何识别欺诈行为",
        "洗钱的典型特征有哪些",
        "企业贷款违约的处理方式"
    ]
    
    print("\n2. 单collection检索测试:")
    for query in test_queries:
        print(f"\n   查询: '{query}'")
        start_time = time.time()
        results = retriever.retrieve(
            query=query,
            collection_name="risk_cases",
            top_k=3
        )
        elapsed = time.time() - start_time
        
        print(f"   返回结果: {len(results)} 条")
        print(f"   耗时: {elapsed:.3f}s")
        
        for i, result in enumerate(results, 1):
            print(f"   [{i}] 相似度: {result.score:.4f}")
            print(f"       内容: {result.content[:80]}...")
    
    print("\n3. 多collection检索测试:")
    query = "风险控制措施"
    print(f"\n   查询: '{query}'")
    start_time = time.time()
    multi_results = retriever.retrieve_multi_collection(
        query=query,
        top_k_per_collection=2
    )
    elapsed = time.time() - start_time
    
    print(f"   耗时: {elapsed:.3f}s")
    for collection_name, results in multi_results.items():
        print(f"\n   {collection_name}: {len(results)} 条结果")
        for result in results[:2]:
            print(f"      - 相似度: {result.score:.4f}")
    
    print("\n4. 元数据过滤测试:")
    query = "欺诈案例"
    print(f"\n   查询: '{query}' (过滤案例类型: 欺诈)")
    results = retriever.retrieve_by_case_type(
        query=query,
        case_type="欺诈",
        top_k=3
    )
    print(f"   返回结果: {len(results)} 条")
    for result in results[:2]:
        print(f"   - 案例类型: {result.metadata.get('case_type', 'N/A')}")
        print(f"     内容: {result.content[:60]}...")
    
    print("\n5. 合并检索测试:")
    query = "风险识别方法"
    print(f"\n   查询: '{query}'")
    results = retriever.retrieve_merged(
        query=query,
        top_k_per_collection=2,
        final_top_k=5
    )
    print(f"   返回结果: {len(results)} 条")
    for i, result in enumerate(results, 1):
        print(f"   [{i}] 来源: {result.collection_name}")
        print(f"       相似度: {result.score:.4f}")


def test_hybrid_retriever():
    """测试混合检索功能"""
    print_separator("测试混合检索功能")
    
    hybrid_retriever = create_hybrid_retriever(
        top_k=5,
        fusion_method="rrf",
        rrf_k=60
    )
    
    print("\n1. 构建BM25索引:")
    hybrid_retriever.build_all_bm25_indexes()
    
    test_queries = [
        "如何防范欺诈风险",
        "洗钱案例分析",
        "信用违约处理流程"
    ]
    
    print("\n2. 向量检索 vs BM25检索对比:")
    for query in test_queries:
        print(f"\n   查询: '{query}'")
        
        print("\n   向量检索结果:")
        vector_results = hybrid_retriever.retrieve_vector(
            query=query,
            collection_name="risk_cases",
            top_k=3
        )
        for i, result in enumerate(vector_results, 1):
            print(f"      [{i}] 相似度: {result.score:.4f} - {result.content[:50]}...")
        
        print("\n   BM25检索结果:")
        bm25_results = hybrid_retriever.retrieve_bm25(
            query=query,
            collection_name="risk_cases",
            top_k=3
        )
        for i, (idx, score, content, metadata) in enumerate(bm25_results, 1):
            print(f"      [{i}] BM25分数: {score:.4f} - {content[:50]}...")
    
    print("\n3. RRF融合测试:")
    query = "欺诈风险识别方法"
    print(f"\n   查询: '{query}'")
    
    hybrid_retriever.set_fusion_method("rrf", rrf_k=60)
    start_time = time.time()
    rrf_results = hybrid_retriever.retrieve_hybrid(
        query=query,
        collection_name="risk_cases",
        top_k=5
    )
    elapsed = time.time() - start_time
    
    print(f"   耗时: {elapsed:.3f}s")
    print(f"   返回结果: {len(rrf_results)} 条")
    for i, result in enumerate(rrf_results, 1):
        print(f"   [{i}] 综合分数: {result.combined_score:.4f}")
        print(f"       向量分数: {result.vector_score:.4f} (排名#{result.rank_vector})")
        print(f"       BM25分数: {result.bm25_score:.4f} (排名#{result.rank_bm25})")
        print(f"       内容: {result.content[:60]}...")
    
    print("\n4. 加权融合测试:")
    print(f"\n   查询: '{query}'")
    
    hybrid_retriever.set_fusion_method(
        "weighted",
        vector_weight=0.6,
        bm25_weight=0.4
    )
    start_time = time.time()
    weighted_results = hybrid_retriever.retrieve_hybrid(
        query=query,
        collection_name="risk_cases",
        top_k=5
    )
    elapsed = time.time() - start_time
    
    print(f"   耗时: {elapsed:.3f}s")
    print(f"   返回结果: {len(weighted_results)} 条")
    for i, result in enumerate(weighted_results, 1):
        print(f"   [{i}] 综合分数: {result.combined_score:.4f}")
        print(f"       向量分数: {result.vector_score:.4f}")
        print(f"       BM25分数: {result.bm25_score:.4f}")
    
    print("\n5. 多collection混合检索测试:")
    query = "风险控制措施"
    print(f"\n   查询: '{query}'")
    
    results = hybrid_retriever.retrieve_hybrid_multi_collection(
        query=query,
        top_k_per_collection=2,
        final_top_k=5
    )
    
    print(f"   返回结果: {len(results)} 条")
    for i, result in enumerate(results, 1):
        print(f"   [{i}] 来源: {result.collection_name}")
        print(f"       综合分数: {result.combined_score:.4f}")
        print(f"       内容: {result.content[:60]}...")


def compare_retrieval_strategies():
    """对比不同检索策略"""
    print_separator("检索策略对比")
    
    vector_retriever = create_retriever(top_k=5, similarity_threshold=0.3)
    hybrid_retriever = create_hybrid_retriever(top_k=5, fusion_method="rrf")
    hybrid_retriever.build_all_bm25_indexes()
    
    test_queries = [
        "如何识别和处理欺诈行为",
        "洗钱风险的典型特征和防范措施",
        "企业信用违约后的处置流程"
    ]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        print("-" * 50)
        
        print("\n【向量检索】")
        start_time = time.time()
        vector_results = vector_retriever.retrieve(
            query=query,
            collection_name="risk_cases",
            top_k=5
        )
        vector_time = time.time() - start_time
        
        for i, result in enumerate(vector_results[:3], 1):
            print(f"   [{i}] 相似度: {result.score:.4f}")
            print(f"       {result.content[:70]}...")
        print(f"   耗时: {vector_time:.3f}s")
        
        print("\n【BM25检索】")
        start_time = time.time()
        bm25_results = hybrid_retriever.retrieve_bm25(
            query=query,
            collection_name="risk_cases",
            top_k=5
        )
        bm25_time = time.time() - start_time
        
        for i, (idx, score, content, metadata) in enumerate(bm25_results[:3], 1):
            print(f"   [{i}] BM25分数: {score:.4f}")
            print(f"       {content[:70]}...")
        print(f"   耗时: {bm25_time:.3f}s")
        
        print("\n【RRF混合检索】")
        start_time = time.time()
        rrf_results = hybrid_retriever.retrieve_hybrid(
            query=query,
            collection_name="risk_cases",
            top_k=5
        )
        rrf_time = time.time() - start_time
        
        for i, result in enumerate(rrf_results[:3], 1):
            print(f"   [{i}] 综合分数: {result.combined_score:.4f}")
            print(f"       向量排名#{result.rank_vector}, BM25排名#{result.rank_bm25}")
            print(f"       {result.content[:70]}...")
        print(f"   耗时: {rrf_time:.3f}s")
        
        hybrid_retriever.set_fusion_method("weighted", vector_weight=0.6, bm25_weight=0.4)
        print("\n【加权混合检索】(向量0.6 + BM25 0.4)")
        start_time = time.time()
        weighted_results = hybrid_retriever.retrieve_hybrid(
            query=query,
            collection_name="risk_cases",
            top_k=5
        )
        weighted_time = time.time() - start_time
        
        for i, result in enumerate(weighted_results[:3], 1):
            print(f"   [{i}] 综合分数: {result.combined_score:.4f}")
            print(f"       {result.content[:70]}...")
        print(f"   耗时: {weighted_time:.3f}s")


def retrieval_examples():
    """检索示例展示"""
    print_separator("检索示例展示")
    
    hybrid_retriever = create_hybrid_retriever(top_k=5, fusion_method="rrf")
    hybrid_retriever.build_all_bm25_indexes()
    
    examples = [
        {
            "query": "某客户通过伪造收入证明申请贷款，如何识别和处理？",
            "description": "欺诈风险识别场景"
        },
        {
            "query": "企业账户频繁大额资金转入转出，如何判断是否涉及洗钱？",
            "description": "洗钱风险识别场景"
        },
        {
            "query": "贷款客户连续逾期，企业主营业务下滑，如何处置？",
            "description": "信用违约处置场景"
        }
    ]
    
    for example in examples:
        query = example["query"]
        print(f"\n场景: {example['description']}")
        print(f"查询: {query}")
        print("-" * 50)
        
        results = hybrid_retriever.retrieve_hybrid_multi_collection(
            query=query,
            top_k_per_collection=3,
            final_top_k=5
        )
        
        print("\n检索结果:")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] 来源: {result.collection_name}")
            print(f"    综合分数: {result.combined_score:.4f}")
            print(f"    内容:")
            print(f"    {result.content}")
            if result.metadata:
                print(f"    元数据: {result.metadata}")


def main():
    """主测试函数"""
    print_separator("RAG检索模块测试")
    print("\n本测试脚本将测试以下内容:")
    print("1. Embedding向量化功能")
    print("2. 向量检索功能")
    print("3. 混合检索功能")
    print("4. 检索策略对比")
    print("5. 检索示例展示")
    
    try:
        test_embedding()
    except Exception as e:
        print(f"\nEmbedding测试失败: {e}")
        print("请确保已配置正确的API密钥")
    
    try:
        test_vector_retriever()
    except Exception as e:
        print(f"\n向量检索测试失败: {e}")
        print("请确保向量库已构建")
    
    try:
        test_hybrid_retriever()
    except Exception as e:
        print(f"\n混合检索测试失败: {e}")
    
    try:
        compare_retrieval_strategies()
    except Exception as e:
        print(f"\n检索策略对比测试失败: {e}")
    
    try:
        retrieval_examples()
    except Exception as e:
        print(f"\n检索示例展示失败: {e}")
    
    print_separator("测试完成")


if __name__ == "__main__":
    main()
