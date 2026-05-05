"""
混合检索模块 - BM25 + 向量检索融合实现

提供混合检索功能，包括：
- BM25稀疏检索
- 向量稠密检索
- RRF融合策略
- 加权融合策略
"""

import math
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
from abc import ABC, abstractmethod

from models.rag.retriever import VectorRetriever, SearchResult, RetrievalConfig
from models.rag.embedding import DoubaoEmbedding


@dataclass
class HybridResult:
    """混合检索结果"""
    id: str
    content: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    collection_name: str = ""
    rank_vector: int = 0
    rank_bm25: int = 0


@dataclass
class HybridConfig:
    """混合检索配置"""
    top_k: int = 5
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    fusion_method: str = "rrf"
    rrf_k: int = 60
    similarity_threshold: float = 0.0


class BM25:
    """
    BM25稀疏检索实现
    
    实现Okapi BM25算法进行文本检索
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        初始化BM25
        
        Args:
            k1: 词频饱和参数
            b: 文档长度归一化参数
            epsilon: IDF下限调整参数
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 词项列表
        """
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def fit(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        构建BM25索引
        
        Args:
            documents: 文档列表
            doc_ids: 文档ID列表
            metadatas: 元数据列表
        """
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        self.doc_metadatas = metadatas or [{} for _ in documents]
        self.corpus_size = len(documents)
        
        total_len = 0
        for doc in documents:
            tokens = self._tokenize(doc)
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(dict(term_freqs))
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            
            for term in term_freqs:
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                self.doc_freqs[term] += 1
        
        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
        self._calc_idf()
    
    def _calc_idf(self):
        """计算IDF值"""
        idf_sum = 0
        negative_idfs = []
        
        for term, freq in self.doc_freqs.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(term)
        
        avg_idf = idf_sum / len(self.idf) if self.idf else 0
        eps = self.epsilon * avg_idf
        
        for term in negative_idfs:
            self.idf[term] = eps
    
    def get_scores(self, query: str) -> List[float]:
        """
        计算查询与所有文档的BM25分数
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: BM25分数列表
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc_idx in range(self.corpus_size):
            score = 0.0
            doc_term_freq = self.doc_term_freqs[doc_idx]
            doc_len = self.doc_len[doc_idx]
            
            for term in query_tokens:
                if term not in self.idf:
                    continue
                
                if term not in doc_term_freq:
                    continue
                
                freq = doc_term_freq[term]
                idf = self.idf[term]
                
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
            
            scores.append(score)
        
        return scores
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """
        搜索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Tuple[int, float, str, Dict]]: (文档索引, 分数, 文档内容, 元数据)
        """
        scores = self.get_scores(query)
        
        scored_docs = [
            (idx, score, self.documents[idx], self.doc_metadatas[idx])
            for idx in range(len(scores))
        ]
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


class FusionStrategy(ABC):
    """融合策略基类"""
    
    @abstractmethod
    def fuse(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[Tuple[int, float, str, Dict[str, Any]]],
        bm25_doc_ids: List[str],
        top_k: int
    ) -> List[HybridResult]:
        """融合向量检索和BM25检索结果"""
        pass


class RRFFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) 融合策略
    
    公式: RRF(d) = sum(1 / (k + rank(d)))
    """
    
    def __init__(self, k: int = 60):
        """
        初始化RRF融合
        
        Args:
            k: RRF参数
        """
        self.k = k
    
    def fuse(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[Tuple[int, float, str, Dict[str, Any]]],
        bm25_doc_ids: List[str],
        top_k: int
    ) -> List[HybridResult]:
        """RRF融合"""
        rrf_scores: Dict[str, Dict[str, Any]] = {}
        
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.id
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "vector_score": result.score,
                    "bm25_score": 0.0,
                    "rank_vector": rank,
                    "rank_bm25": 0,
                    "collection_name": result.collection_name,
                    "rrf_vector": 1 / (self.k + rank),
                    "rrf_bm25": 0.0
                }
            else:
                rrf_scores[doc_id]["vector_score"] = result.score
                rrf_scores[doc_id]["rank_vector"] = rank
                rrf_scores[doc_id]["rrf_vector"] = 1 / (self.k + rank)
        
        for rank, (idx, score, content, metadata) in enumerate(bm25_results, 1):
            doc_id = bm25_doc_ids[idx] if idx < len(bm25_doc_ids) else str(idx)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "content": content,
                    "metadata": metadata,
                    "vector_score": 0.0,
                    "bm25_score": score,
                    "rank_vector": 0,
                    "rank_bm25": rank,
                    "collection_name": "",
                    "rrf_vector": 0.0,
                    "rrf_bm25": 1 / (self.k + rank)
                }
            else:
                rrf_scores[doc_id]["bm25_score"] = score
                rrf_scores[doc_id]["rank_bm25"] = rank
                rrf_scores[doc_id]["rrf_bm25"] = 1 / (self.k + rank)
        
        for doc_id, data in rrf_scores.items():
            data["combined_score"] = data["rrf_vector"] + data["rrf_bm25"]
        
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )
        
        hybrid_results = []
        for doc_id, data in sorted_results[:top_k]:
            hybrid_results.append(HybridResult(
                id=doc_id,
                content=data["content"],
                metadata=data["metadata"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=data["combined_score"],
                collection_name=data["collection_name"],
                rank_vector=data["rank_vector"],
                rank_bm25=data["rank_bm25"]
            ))
        
        return hybrid_results


class WeightedFusion(FusionStrategy):
    """
    加权融合策略
    
    公式: score = w1 * vector_score + w2 * bm25_score
    """
    
    def __init__(self, vector_weight: float = 0.5, bm25_weight: float = 0.5):
        """
        初始化加权融合
        
        Args:
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def _normalize_scores(
        self,
        scores: List[float]
    ) -> List[float]:
        """归一化分数到[0, 1]"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def fuse(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[Tuple[int, float, str, Dict[str, Any]]],
        bm25_doc_ids: List[str],
        top_k: int
    ) -> List[HybridResult]:
        """加权融合"""
        vector_scores = [r.score for r in vector_results]
        bm25_scores = [r[1] for r in bm25_results]
        
        norm_vector_scores = self._normalize_scores(vector_scores)
        norm_bm25_scores = self._normalize_scores(bm25_scores)
        
        combined: Dict[str, Dict[str, Any]] = {}
        
        for rank, (result, norm_score) in enumerate(zip(vector_results, norm_vector_scores)):
            doc_id = result.id
            if doc_id not in combined:
                combined[doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "vector_score": result.score,
                    "norm_vector_score": norm_score,
                    "bm25_score": 0.0,
                    "norm_bm25_score": 0.0,
                    "rank_vector": rank + 1,
                    "rank_bm25": 0,
                    "collection_name": result.collection_name
                }
            else:
                combined[doc_id]["vector_score"] = result.score
                combined[doc_id]["norm_vector_score"] = norm_score
                combined[doc_id]["rank_vector"] = rank + 1
        
        for rank, ((idx, score, content, metadata), norm_score) in enumerate(
            zip(bm25_results, norm_bm25_scores)
        ):
            doc_id = bm25_doc_ids[idx] if idx < len(bm25_doc_ids) else str(idx)
            if doc_id not in combined:
                combined[doc_id] = {
                    "content": content,
                    "metadata": metadata,
                    "vector_score": 0.0,
                    "norm_vector_score": 0.0,
                    "bm25_score": score,
                    "norm_bm25_score": norm_score,
                    "rank_vector": 0,
                    "rank_bm25": rank + 1,
                    "collection_name": ""
                }
            else:
                combined[doc_id]["bm25_score"] = score
                combined[doc_id]["norm_bm25_score"] = norm_score
                combined[doc_id]["rank_bm25"] = rank + 1
        
        for doc_id, data in combined.items():
            data["combined_score"] = (
                self.vector_weight * data["norm_vector_score"] +
                self.bm25_weight * data["norm_bm25_score"]
            )
        
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )
        
        hybrid_results = []
        for doc_id, data in sorted_results[:top_k]:
            hybrid_results.append(HybridResult(
                id=doc_id,
                content=data["content"],
                metadata=data["metadata"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=data["combined_score"],
                collection_name=data["collection_name"],
                rank_vector=data["rank_vector"],
                rank_bm25=data["rank_bm25"]
            ))
        
        return hybrid_results


class HybridRetriever:
    """
    混合检索器
    
    结合BM25稀疏检索和向量稠密检索
    支持RRF和加权融合策略
    """
    
    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        config: Optional[HybridConfig] = None
    ):
        """
        初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器实例
            config: 混合检索配置
        """
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.config = config or HybridConfig()
        
        self.bm25_indexes: Dict[str, BM25] = {}
        self.bm25_doc_ids: Dict[str, List[str]] = {}
        
        self._fusion_strategy = self._get_fusion_strategy()
    
    def _get_fusion_strategy(self) -> FusionStrategy:
        """获取融合策略"""
        if self.config.fusion_method == "rrf":
            return RRFFusion(k=self.config.rrf_k)
        elif self.config.fusion_method == "weighted":
            return WeightedFusion(
                vector_weight=self.config.vector_weight,
                bm25_weight=self.config.bm25_weight
            )
        else:
            return RRFFusion(k=self.config.rrf_k)
    
    def build_bm25_index(
        self,
        collection_name: str,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        为指定collection构建BM25索引
        
        Args:
            collection_name: collection名称
            documents: 文档列表
            doc_ids: 文档ID列表
            metadatas: 元数据列表
        """
        bm25 = BM25()
        bm25.fit(documents, doc_ids, metadatas)
        
        self.bm25_indexes[collection_name] = bm25
        self.bm25_doc_ids[collection_name] = doc_ids or [str(i) for i in range(len(documents))]
    
    def build_bm25_from_collection(
        self,
        collection_name: str
    ):
        """
        从Chroma collection构建BM25索引
        
        Args:
            collection_name: collection名称
        """
        collection = self.vector_retriever.get_collection(collection_name)
        if not collection:
            print(f"Collection '{collection_name}' 不存在")
            return
        
        results = collection.get(include=["documents", "metadatas"])
        
        documents = results.get("documents", [])
        doc_ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        
        self.build_bm25_index(collection_name, documents, doc_ids, metadatas)
        print(f"BM25索引构建完成: {collection_name}, 文档数: {len(documents)}")
    
    def build_all_bm25_indexes(self):
        """为所有collection构建BM25索引"""
        for collection_name in self.vector_retriever.list_collections():
            try:
                self.build_bm25_from_collection(collection_name)
            except Exception as e:
                print(f"构建BM25索引失败 '{collection_name}': {e}")
    
    def retrieve_vector(
        self,
        query: str,
        collection_name: str = "risk_cases",
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        执行向量检索
        
        Args:
            query: 查询文本
            collection_name: collection名称
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 向量检索结果
        """
        top_k = top_k or self.config.top_k
        return self.vector_retriever.retrieve(
            query=query,
            collection_name=collection_name,
            top_k=top_k
        )
    
    def retrieve_bm25(
        self,
        query: str,
        collection_name: str = "risk_cases",
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """
        执行BM25检索
        
        Args:
            query: 查询文本
            collection_name: collection名称
            top_k: 返回结果数量
            
        Returns:
            List[Tuple]: BM25检索结果
        """
        top_k = top_k or self.config.top_k
        
        if collection_name not in self.bm25_indexes:
            self.build_bm25_from_collection(collection_name)
        
        bm25 = self.bm25_indexes.get(collection_name)
        if not bm25:
            return []
        
        return bm25.search(query, top_k)
    
    def retrieve_hybrid(
        self,
        query: str,
        collection_name: str = "risk_cases",
        top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None
    ) -> List[HybridResult]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            collection_name: collection名称
            top_k: 最终返回结果数量
            vector_top_k: 向量检索结果数量
            bm25_top_k: BM25检索结果数量
            
        Returns:
            List[HybridResult]: 混合检索结果
        """
        top_k = top_k or self.config.top_k
        vector_top_k = vector_top_k or top_k * 2
        bm25_top_k = bm25_top_k or top_k * 2
        
        vector_results = self.retrieve_vector(query, collection_name, vector_top_k)
        
        bm25_results = self.retrieve_bm25(query, collection_name, bm25_top_k)
        
        bm25_doc_ids = self.bm25_doc_ids.get(collection_name, [])
        
        hybrid_results = self._fusion_strategy.fuse(
            vector_results,
            bm25_results,
            bm25_doc_ids,
            top_k
        )
        
        if self.config.similarity_threshold > 0:
            hybrid_results = [
                r for r in hybrid_results
                if r.combined_score >= self.config.similarity_threshold
            ]
        
        return hybrid_results
    
    def retrieve_hybrid_multi_collection(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        top_k_per_collection: int = 3,
        final_top_k: int = 5
    ) -> List[HybridResult]:
        """
        多collection混合检索
        
        Args:
            query: 查询文本
            collection_names: collection名称列表
            top_k_per_collection: 每个collection返回的结果数量
            final_top_k: 最终返回的结果数量
            
        Returns:
            List[HybridResult]: 混合检索结果
        """
        collection_names = collection_names or VectorRetriever.DEFAULT_COLLECTION_NAMES
        
        all_results = []
        for name in collection_names:
            try:
                results = self.retrieve_hybrid(
                    query=query,
                    collection_name=name,
                    top_k=top_k_per_collection
                )
                all_results.extend(results)
            except Exception as e:
                print(f"混合检索失败 '{name}': {e}")
        
        all_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return all_results[:final_top_k]
    
    def set_fusion_method(
        self,
        method: str,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        rrf_k: Optional[int] = None
    ):
        """
        设置融合方法
        
        Args:
            method: 融合方法 ('rrf' 或 'weighted')
            vector_weight: 向量权重（仅weighted方法）
            bm25_weight: BM25权重（仅weighted方法）
            rrf_k: RRF参数（仅rrf方法）
        """
        self.config.fusion_method = method
        
        if vector_weight is not None:
            self.config.vector_weight = vector_weight
        if bm25_weight is not None:
            self.config.bm25_weight = bm25_weight
        if rrf_k is not None:
            self.config.rrf_k = rrf_k
        
        self._fusion_strategy = self._get_fusion_strategy()
    
    def format_results(
        self,
        results: List[HybridResult],
        include_scores: bool = True
    ) -> str:
        """
        格式化混合检索结果
        
        Args:
            results: 混合检索结果列表
            include_scores: 是否包含分数信息
            
        Returns:
            str: 格式化后的字符串
        """
        formatted = []
        for i, result in enumerate(results, 1):
            text = f"[{i}] {result.content}"
            if include_scores:
                text += f"\n    综合分数: {result.combined_score:.4f}"
                text += f" (向量: {result.vector_score:.4f}, BM25: {result.bm25_score:.4f})"
                text += f"\n    排名: 向量#{result.rank_vector}, BM25#{result.rank_bm25}"
            formatted.append(text)
        
        return "\n\n".join(formatted)


def create_hybrid_retriever(
    top_k: int = 5,
    fusion_method: str = "rrf",
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rrf_k: int = 60
) -> HybridRetriever:
    """
    创建混合检索器实例
    
    Args:
        top_k: 返回结果数量
        fusion_method: 融合方法
        vector_weight: 向量权重
        bm25_weight: BM25权重
        rrf_k: RRF参数
        
    Returns:
        HybridRetriever: 混合检索器实例
    """
    config = HybridConfig(
        top_k=top_k,
        fusion_method=fusion_method,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k
    )
    return HybridRetriever(config=config)
