"""
Embedding模块 - 豆包Embedding向量化实现

提供文本向量化功能，包括：
- 单文本和批量文本向量化
- 缓存机制减少API调用
- 错误处理和重试机制
"""

import os
import json
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

from volcengine.maas import MaasException

from utils.volcengine_api import VolcEngineAPI, TokenUsage, CostEstimate


@dataclass
class EmbeddingCache:
    """Embedding缓存配置"""
    cache_dir: str = "./cache/embeddings"
    enabled: bool = True
    max_memory_cache_size: int = 1000


@dataclass
class EmbeddingResult:
    """向量化结果"""
    embeddings: List[List[float]]
    token_usage: TokenUsage
    cost: CostEstimate
    cached_count: int = 0
    api_call_count: int = 0


class DoubaoEmbedding:
    """
    豆包Embedding向量化类
    
    使用豆包Embedding API（doubao-embedding-vision-251215）进行文本向量化
    支持缓存机制减少API调用
    """
    
    MODEL_NAME = "doubao-embedding-vision-251215"
    EMBEDDING_DIMENSION = 2048
    MAX_BATCH_SIZE = 20
    MAX_TEXT_LENGTH = 8000
    
    def __init__(
        self,
        cache_config: Optional[EmbeddingCache] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        初始化Embedding模型
        
        Args:
            cache_config: 缓存配置
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.api = VolcEngineAPI(
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        
        self.cache_config = cache_config or EmbeddingCache()
        
        if self.cache_config.enabled:
            Path(self.cache_config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self._memory_cache: Dict[str, List[float]] = {}
        self._cache_order: List[str] = []
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        normalized_text = text.strip().lower()
        return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_config.cache_dir, f"{cache_key}.json")
    
    def _load_from_memory_cache(self, cache_key: str) -> Optional[List[float]]:
        """从内存缓存加载"""
        if cache_key in self._memory_cache:
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._memory_cache[cache_key]
        return None
    
    def _save_to_memory_cache(self, cache_key: str, embedding: List[float]):
        """保存到内存缓存"""
        if len(self._memory_cache) >= self.cache_config.max_memory_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = embedding
        self._cache_order.append(cache_key)
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[List[float]]:
        """从磁盘缓存加载"""
        cache_file = self._get_cache_file_path(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('embedding')
            except Exception:
                pass
        return None
    
    def _save_to_disk_cache(self, cache_key: str, embedding: List[float]):
        """保存到磁盘缓存"""
        cache_file = self._get_cache_file_path(cache_key)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'embedding': embedding,
                    'model': self.MODEL_NAME,
                    'timestamp': time.time()
                }, f)
        except Exception:
            pass
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """获取缓存的embedding"""
        if not self.cache_config.enabled:
            return None
        
        cache_key = self._get_cache_key(text)
        
        embedding = self._load_from_memory_cache(cache_key)
        if embedding is not None:
            return embedding
        
        embedding = self._load_from_disk_cache(cache_key)
        if embedding is not None:
            self._save_to_memory_cache(cache_key, embedding)
            return embedding
        
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """缓存embedding"""
        if not self.cache_config.enabled:
            return
        
        cache_key = self._get_cache_key(text)
        self._save_to_memory_cache(cache_key, embedding)
        self._save_to_disk_cache(cache_key, embedding)
    
    def _truncate_text(self, text: str) -> str:
        """截断过长的文本"""
        if len(text) > self.MAX_TEXT_LENGTH:
            return text[:self.MAX_TEXT_LENGTH]
        return text
    
    def embed_single(
        self,
        text: str,
        use_cache: bool = True
    ) -> Tuple[List[float], TokenUsage, CostEstimate]:
        """
        单文本向量化
        
        Args:
            text: 待向量化的文本
            use_cache: 是否使用缓存
            
        Returns:
            Tuple[List[float], TokenUsage, CostEstimate]: 向量、Token使用量、成本估算
        """
        text = self._truncate_text(text)
        
        if use_cache:
            cached = self._get_cached_embedding(text)
            if cached is not None:
                return cached, TokenUsage(), CostEstimate()
        
        embedding, token_usage, cost = self.api.call_embedding(text)
        
        if embedding and use_cache:
            self._cache_embedding(text, embedding)
        
        return embedding, token_usage, cost
    
    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: int = 10
    ) -> EmbeddingResult:
        """
        批量文本向量化
        
        Args:
            texts: 待向量化的文本列表
            use_cache: 是否使用缓存
            batch_size: API调用批次大小
            
        Returns:
            EmbeddingResult: 向量化结果
        """
        all_embeddings: List[List[float]] = []
        total_token_usage = TokenUsage()
        total_cost = CostEstimate()
        cached_count = 0
        api_call_count = 0
        
        texts_to_embed: List[Tuple[int, str]] = []
        cached_embeddings: Dict[int, List[float]] = {}
        
        for idx, text in enumerate(texts):
            text = self._truncate_text(text)
            
            if use_cache:
                cached = self._get_cached_embedding(text)
                if cached is not None:
                    cached_embeddings[idx] = cached
                    cached_count += 1
                    continue
            
            texts_to_embed.append((idx, text))
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            batch_texts = [text for _, text in batch]
            batch_indices = [idx for idx, _ in batch]
            
            embeddings, token_usage, cost = self.api.batch_embedding(batch_texts)
            
            for idx, embedding in zip(batch_indices, embeddings):
                if embedding:
                    cached_embeddings[idx] = embedding
                    if use_cache:
                        self._cache_embedding(texts[idx], embedding)
            
            total_token_usage.prompt_tokens += token_usage.prompt_tokens
            total_token_usage.completion_tokens += token_usage.completion_tokens
            total_token_usage.total_tokens += token_usage.total_tokens
            total_cost.input_cost += cost.input_cost
            total_cost.output_cost += cost.output_cost
            total_cost.total_cost += cost.total_cost
            api_call_count += len(batch_texts)
        
        for idx in range(len(texts)):
            if idx in cached_embeddings:
                all_embeddings.append(cached_embeddings[idx])
            else:
                all_embeddings.append([])
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            token_usage=total_token_usage,
            cost=total_cost,
            cached_count=cached_count,
            api_call_count=api_call_count
        )
    
    def embed_query(self, query: str) -> List[float]:
        """
        向量化查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 查询向量
        """
        embedding, _, _ = self.embed_single(query, use_cache=True)
        return embedding
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        向量化文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            List[List[float]]: 文档向量列表
        """
        result = self.embed_batch(documents, use_cache=True)
        return result.embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """获取API调用统计"""
        return self.api.get_stats()
    
    def reset_stats(self):
        """重置统计"""
        self.api.reset_stats()
    
    def clear_cache(self):
        """清空缓存"""
        self._memory_cache.clear()
        self._cache_order.clear()
        
        if os.path.exists(self.cache_config.cache_dir):
            import shutil
            shutil.rmtree(self.cache_config.cache_dir)
            Path(self.cache_config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        disk_cache_count = 0
        if os.path.exists(self.cache_config.cache_dir):
            disk_cache_count = len([f for f in os.listdir(self.cache_config.cache_dir) 
                                   if f.endswith('.json')])
        
        return {
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_size": disk_cache_count,
            "cache_enabled": self.cache_config.enabled
        }


def create_embedding_model(
    cache_enabled: bool = True,
    cache_dir: str = "./cache/embeddings"
) -> DoubaoEmbedding:
    """
    创建Embedding模型实例
    
    Args:
        cache_enabled: 是否启用缓存
        cache_dir: 缓存目录
        
    Returns:
        DoubaoEmbedding: Embedding模型实例
    """
    cache_config = EmbeddingCache(
        cache_dir=cache_dir,
        enabled=cache_enabled
    )
    return DoubaoEmbedding(cache_config=cache_config)
