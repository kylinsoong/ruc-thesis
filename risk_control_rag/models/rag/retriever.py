"""
向量检索模块 - Chroma向量库检索实现

提供向量检索功能，包括：
- Top-K检索
- 相似度阈值过滤
- 元数据过滤
- 多collection检索
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from config.config import DATA_CONFIG, RAG_CONFIG
from models.rag.embedding import DoubaoEmbedding, create_embedding_model


@dataclass
class SearchResult:
    """检索结果"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    collection_name: str


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    distance_metric: str = "cosine"


class VectorRetriever:
    """
    向量检索器
    
    使用Chroma向量库实现向量检索
    支持Top-K检索、相似度阈值过滤、元数据过滤
    """
    
    DEFAULT_COLLECTION_NAMES = ["risk_cases", "regulations", "industry_knowledge"]
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[DoubaoEmbedding] = None,
        config: Optional[RetrievalConfig] = None
    ):
        """
        初始化向量检索器
        
        Args:
            persist_dir: Chroma持久化目录
            embedding_model: Embedding模型实例
            config: 检索配置
        """
        self.persist_dir = persist_dir or os.path.join(
            DATA_CONFIG["knowledge_base_dir"], "chroma_db"
        )
        
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_model = embedding_model or create_embedding_model()
        self.config = config or RetrievalConfig(
            top_k=RAG_CONFIG.get("top_k", 5),
            similarity_threshold=RAG_CONFIG.get("similarity_threshold", 0.7)
        )
        
        self.collections: Dict[str, chromadb.Collection] = {}
    
    def get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """
        获取或加载collection
        
        Args:
            name: collection名称
            
        Returns:
            chromadb.Collection: collection实例
        """
        if name not in self.collections:
            try:
                self.collections[name] = self.client.get_collection(name=name)
            except Exception as e:
                print(f"获取collection '{name}' 失败: {e}")
                return None
        return self.collections[name]
    
    def list_collections(self) -> List[str]:
        """列出所有collection"""
        return [c.name for c in self.client.list_collections()]
    
    def get_collection_count(self, name: str) -> int:
        """获取collection中的文档数量"""
        collection = self.get_collection(name)
        if collection:
            return collection.count()
        return 0
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        将距离转换为相似度
        
        Args:
            distance: 距离值
            
        Returns:
            float: 相似度值 (0-1)
        """
        if self.config.distance_metric == "cosine":
            return 1 - distance
        elif self.config.distance_metric == "l2":
            return 1 / (1 + distance)
        else:
            return 1 - distance
    
    def _filter_by_threshold(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        根据相似度阈值过滤结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            List[SearchResult]: 过滤后的结果
        """
        return [
            r for r in results
            if r.score >= self.config.similarity_threshold
        ]
    
    def retrieve(
        self,
        query: str,
        collection_name: str = "risk_cases",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        执行向量检索
        
        Args:
            query: 查询文本
            collection_name: collection名称
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return []
        
        top_k = top_k or self.config.top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        query_embedding = self.embedding_model.embed_query(query)
        if not query_embedding:
            print("获取查询向量失败")
            return []
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"查询失败: {e}")
            return []
        
        search_results = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = self._distance_to_similarity(distance)
                
                if score >= similarity_threshold:
                    search_results.append(SearchResult(
                        id=results["ids"][0][i] if results["ids"] else str(i),
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=score,
                        collection_name=collection_name
                    ))
        
        return search_results
    
    def retrieve_multi_collection(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        top_k_per_collection: int = 3,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, List[SearchResult]]:
        """
        多collection检索
        
        Args:
            query: 查询文本
            collection_names: collection名称列表
            top_k_per_collection: 每个collection返回的结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            Dict[str, List[SearchResult]]: 各collection的检索结果
        """
        collection_names = collection_names or self.DEFAULT_COLLECTION_NAMES
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        results = {}
        for name in collection_names:
            try:
                collection_results = self.retrieve(
                    query=query,
                    collection_name=name,
                    top_k=top_k_per_collection,
                    similarity_threshold=similarity_threshold
                )
                results[name] = collection_results
            except Exception as e:
                print(f"检索collection '{name}' 失败: {e}")
                results[name] = []
        
        return results
    
    def retrieve_merged(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        top_k_per_collection: int = 3,
        final_top_k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        多collection检索并合并结果
        
        Args:
            query: 查询文本
            collection_names: collection名称列表
            top_k_per_collection: 每个collection返回的结果数量
            final_top_k: 最终返回的结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            List[SearchResult]: 合并后的检索结果
        """
        multi_results = self.retrieve_multi_collection(
            query=query,
            collection_names=collection_names,
            top_k_per_collection=top_k_per_collection,
            similarity_threshold=similarity_threshold
        )
        
        all_results = []
        for collection_results in multi_results.values():
            all_results.extend(collection_results)
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:final_top_k]
    
    def retrieve_by_case_type(
        self,
        query: str,
        case_type: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        按案例类型检索
        
        Args:
            query: 查询文本
            case_type: 案例类型（如"欺诈"、"洗钱"、"信用违约"）
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果
        """
        return self.retrieve(
            query=query,
            collection_name="risk_cases",
            top_k=top_k,
            where={"case_type": case_type}
        )
    
    def retrieve_by_source(
        self,
        query: str,
        source: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        按规则来源检索
        
        Args:
            query: 查询文本
            source: 规则来源
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果
        """
        return self.retrieve(
            query=query,
            collection_name="regulations",
            top_k=top_k,
            where={"source": source}
        )
    
    def retrieve_by_category(
        self,
        query: str,
        category: str,
        collection_name: str = "regulations",
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        按分类检索
        
        Args:
            query: 查询文本
            category: 分类
            collection_name: collection名称
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果
        """
        return self.retrieve(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            where={"category": category}
        )
    
    def retrieve_with_rerank(
        self,
        query: str,
        collection_name: str = "risk_cases",
        top_k: int = 10,
        final_top_k: int = 5
    ) -> List[SearchResult]:
        """
        检索并重排序
        
        Args:
            query: 查询文本
            collection_name: collection名称
            top_k: 初始检索数量
            final_top_k: 最终返回数量
            
        Returns:
            List[SearchResult]: 重排序后的结果
        """
        results = self.retrieve(
            query=query,
            collection_name=collection_name,
            top_k=top_k
        )
        
        query_terms = set(query.lower().split())
        
        for result in results:
            content_terms = set(result.content.lower().split())
            term_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            result.score = 0.7 * result.score + 0.3 * term_overlap
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:final_top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        stats = {
            "persist_dir": self.persist_dir,
            "collections": {}
        }
        
        for name in self.list_collections():
            collection = self.get_collection(name)
            if collection:
                stats["collections"][name] = {
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
        
        return stats
    
    def format_results(
        self,
        results: List[SearchResult],
        include_score: bool = True,
        include_metadata: bool = False
    ) -> str:
        """
        格式化检索结果为字符串
        
        Args:
            results: 检索结果列表
            include_score: 是否包含相似度分数
            include_metadata: 是否包含元数据
            
        Returns:
            str: 格式化后的字符串
        """
        formatted = []
        for i, result in enumerate(results, 1):
            text = f"[{i}] {result.content}"
            if include_score:
                text += f"\n    相似度: {result.score:.4f}"
            if include_metadata and result.metadata:
                text += f"\n    元数据: {result.metadata}"
            formatted.append(text)
        
        return "\n\n".join(formatted)


def create_retriever(
    persist_dir: Optional[str] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> VectorRetriever:
    """
    创建向量检索器实例
    
    Args:
        persist_dir: Chroma持久化目录
        top_k: 返回结果数量
        similarity_threshold: 相似度阈值
        
    Returns:
        VectorRetriever: 向量检索器实例
    """
    config = RetrievalConfig(
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    return VectorRetriever(persist_dir=persist_dir, config=config)
