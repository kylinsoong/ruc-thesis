"""
RAG风险分析系统 - 完整系统整合模块

整合Embedding、检索、Prompt构建、LLM生成和输出解析模块，
提供完整的风险分析功能。
"""

import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from models.rag.embedding import DoubaoEmbedding, create_embedding_model
from models.rag.retriever import VectorRetriever, SearchResult, create_retriever
from models.rag.hybrid_retriever import HybridRetriever, HybridResult, HybridConfig, create_hybrid_retriever
from models.rag.prompt_builder import (
    PromptBuilder,
    PromptConfig,
    RetrievedKnowledge,
)
from models.rag.generator import RAGGenerator, GenerationConfig, GenerationResult
from models.rag.output_parser import (
    OutputParser,
    ParsedRiskAnalysis,
    RiskLevel,
    ApprovalSuggestion,
)


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RAGSystemConfig:
    """RAG系统配置"""
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 5
    similarity_threshold: float = 0.5
    collection_names: List[str] = field(default_factory=lambda: ["risk_cases", "regulations", "industry_knowledge"])
    
    temperature: float = 0.3
    max_tokens: int = 2048
    template_type: str = "default"
    
    max_context_length: int = 4000
    max_knowledge_items: int = 5
    
    fusion_method: str = "rrf"
    vector_weight: float = 0.5
    bm25_weight: float = 0.5


@dataclass
class RiskAnalysisResult:
    """风险分析结果"""
    customer_id: str = ""
    parsed_result: Optional[ParsedRiskAnalysis] = None
    raw_response: str = ""
    generation_result: Optional[GenerationResult] = None
    retrieval_results: List[Any] = field(default_factory=list)
    knowledge_used: List[RetrievedKnowledge] = field(default_factory=list)
    latency: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "customer_id": self.customer_id,
            "risk_level": self.parsed_result.risk_level_text if self.parsed_result else "未知",
            "risk_score": self.parsed_result.risk_score if self.parsed_result else -1,
            "risk_factors": [f.to_dict() for f in self.parsed_result.risk_factors] if self.parsed_result else [],
            "favorable_factors": [f.to_dict() for f in self.parsed_result.favorable_factors] if self.parsed_result else [],
            "approval_suggestion": self.parsed_result.approval_suggestion_text if self.parsed_result else "待定",
            "credit_limit_suggestion": self.parsed_result.credit_limit_suggestion if self.parsed_result else "",
            "monitoring_indicators": self.parsed_result.monitoring_indicators if self.parsed_result else [],
            "risk_mitigation_measures": self.parsed_result.risk_mitigation_measures if self.parsed_result else [],
            "knowledge_sources": [s.to_dict() for s in self.parsed_result.knowledge_sources] if self.parsed_result else [],
            "raw_response": self.raw_response,
            "latency": self.latency,
            "success": self.success,
            "error_message": self.error_message,
        }
        
        if self.generation_result:
            result["token_usage"] = {
                "prompt_tokens": self.generation_result.prompt_tokens,
                "completion_tokens": self.generation_result.completion_tokens,
                "total_tokens": self.generation_result.total_tokens,
            }
            result["cost"] = self.generation_result.cost
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class RAGRiskSystem:
    """
    RAG风险分析系统
    
    整合以下模块：
    - Embedding模块：文本向量化
    - 检索模块：向量检索和混合检索
    - Prompt构建模块：构建分析提示词
    - LLM生成模块：调用大语言模型生成分析结果
    - 输出解析模块：解析结构化输出
    """
    
    def __init__(
        self,
        config: Optional[RAGSystemConfig] = None,
        persist_dir: Optional[str] = None,
    ):
        """
        初始化RAG风险分析系统
        
        Args:
            config: 系统配置
            persist_dir: 向量库持久化目录
        """
        self.config = config or RAGSystemConfig()
        self.persist_dir = persist_dir
        
        self._init_modules()
        
        self._total_analyses = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
    
    def _init_modules(self):
        """初始化各模块"""
        self.embedding_model = create_embedding_model()
        
        retrieval_config = {
            "top_k": self.config.top_k * 2,
            "similarity_threshold": self.config.similarity_threshold,
        }
        
        self.vector_retriever = create_retriever(
            persist_dir=self.persist_dir,
            **retrieval_config
        )
        
        if self.config.retrieval_strategy == RetrievalStrategy.HYBRID:
            hybrid_config = HybridConfig(
                top_k=self.config.top_k,
                fusion_method=self.config.fusion_method,
                vector_weight=self.config.vector_weight,
                bm25_weight=self.config.bm25_weight,
            )
            self.hybrid_retriever = create_hybrid_retriever(
                top_k=self.config.top_k,
                fusion_method=self.config.fusion_method,
                vector_weight=self.config.vector_weight,
                bm25_weight=self.config.bm25_weight,
            )
            self.hybrid_retriever.vector_retriever = self.vector_retriever
        else:
            self.hybrid_retriever = None
        
        prompt_config = PromptConfig(
            max_context_length=self.config.max_context_length,
            max_knowledge_items=self.config.max_knowledge_items,
            template_type=self.config.template_type,
        )
        self.prompt_builder = PromptBuilder(prompt_config)
        
        generation_config = GenerationConfig(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self.generator = RAGGenerator(generation_config, prompt_config)
        
        self.output_parser = OutputParser(strict_mode=False)
    
    def set_retrieval_strategy(self, strategy: Union[str, RetrievalStrategy]):
        """
        设置检索策略
        
        Args:
            strategy: 检索策略 ('vector' 或 'hybrid')
        """
        if isinstance(strategy, str):
            strategy = RetrievalStrategy(strategy)
        self.config.retrieval_strategy = strategy
    
    def set_template_type(self, template_type: str):
        """
        设置Prompt模板类型
        
        Args:
            template_type: 模板类型 ('default', 'concise', 'detailed')
        """
        self.config.template_type = template_type
        self.prompt_builder.config.template_type = template_type
    
    def _build_query(self, customer_data: Dict[str, Any]) -> str:
        """
        根据客户数据构建检索查询
        
        Args:
            customer_data: 客户数据
            
        Returns:
            str: 检索查询
        """
        query_parts = []
        
        if "purpose" in customer_data:
            purpose = customer_data["purpose"]
            purpose_desc = {
                "A40": "购买新车",
                "A41": "购买二手车",
                "A42": "购买家具设备",
                "A43": "购买收音机电视",
                "A44": "购买家用电器",
                "A45": "维修",
                "A46": "教育",
                "A47": "度假",
                "A48": "再培训",
                "A49": "商业用途",
                "A410": "其他",
            }.get(purpose, purpose)
            query_parts.append(f"贷款目的: {purpose_desc}")
        
        if "credit_history" in customer_data:
            history = customer_data["credit_history"]
            history_desc = {
                "A30": "没有贷款记录",
                "A31": "所有贷款已按时还清",
                "A32": "现有贷款按时还款",
                "A33": "存在延迟还款记录",
                "A34": "存在风险账户",
            }.get(history, history)
            query_parts.append(f"信用历史: {history_desc}")
        
        if "checking_status" in customer_data:
            status = customer_data["checking_status"]
            status_desc = {
                "A11": "账户透支",
                "A12": "账户余额较低",
                "A13": "账户余额充足",
                "A14": "无支票账户",
            }.get(status, status)
            query_parts.append(f"支票账户: {status_desc}")
        
        if "employment" in customer_data:
            emp = customer_data["employment"]
            emp_desc = {
                "A71": "失业",
                "A72": "就业不到1年",
                "A73": "就业1-4年",
                "A74": "就业4-7年",
                "A75": "就业7年以上",
            }.get(emp, emp)
            query_parts.append(f"就业状态: {emp_desc}")
        
        if "credit_amount" in customer_data:
            amount = customer_data["credit_amount"]
            if amount > 10000:
                query_parts.append("高额贷款")
            elif amount > 5000:
                query_parts.append("中等额度贷款")
            else:
                query_parts.append("小额贷款")
        
        if "duration" in customer_data:
            duration = customer_data["duration"]
            if duration > 36:
                query_parts.append("长期贷款")
            elif duration > 12:
                query_parts.append("中期贷款")
            else:
                query_parts.append("短期贷款")
        
        if not query_parts:
            query_parts.append("信贷风险评估")
        
        return " ".join(query_parts)
    
    def retrieve(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
    ) -> List[RetrievedKnowledge]:
        """
        执行检索
        
        Args:
            query: 检索查询
            collection_names: 要检索的collection列表
            
        Returns:
            List[RetrievedKnowledge]: 检索到的知识列表
        """
        collection_names = collection_names or self.config.collection_names
        
        if self.config.retrieval_strategy == RetrievalStrategy.HYBRID and self.hybrid_retriever:
            results = self.hybrid_retriever.retrieve_hybrid_multi_collection(
                query=query,
                collection_names=collection_names,
                top_k_per_collection=self.config.top_k,
                final_top_k=self.config.max_knowledge_items,
            )
            
            knowledge_items = []
            for result in results:
                source_type = result.metadata.get("type", "default")
                if result.collection_name == "risk_cases":
                    source_type = "risk_case"
                elif result.collection_name == "regulations":
                    source_type = "regulation"
                elif result.collection_name == "industry_knowledge":
                    source_type = "industry_knowledge"
                
                knowledge_items.append(RetrievedKnowledge(
                    content=result.content,
                    source_type=source_type,
                    source_id=result.id,
                    relevance_score=result.combined_score,
                    metadata=result.metadata,
                ))
            
            return knowledge_items
        else:
            results = self.vector_retriever.retrieve_merged(
                query=query,
                collection_names=collection_names,
                top_k_per_collection=self.config.top_k,
                final_top_k=self.config.max_knowledge_items,
            )
            
            knowledge_items = []
            for result in results:
                source_type = result.metadata.get("type", "default")
                if result.collection_name == "risk_cases":
                    source_type = "risk_case"
                elif result.collection_name == "regulations":
                    source_type = "regulation"
                elif result.collection_name == "industry_knowledge":
                    source_type = "industry_knowledge"
                
                knowledge_items.append(RetrievedKnowledge(
                    content=result.content,
                    source_type=source_type,
                    source_id=result.id,
                    relevance_score=result.score,
                    metadata=result.metadata,
                ))
            
            return knowledge_items
    
    def analyze(
        self,
        customer_data: Dict[str, Any],
        customer_id: Optional[str] = None,
        template_type: Optional[str] = None,
    ) -> RiskAnalysisResult:
        """
        执行风险分析
        
        Args:
            customer_data: 客户数据
            customer_id: 客户ID
            template_type: Prompt模板类型
            
        Returns:
            RiskAnalysisResult: 风险分析结果
        """
        start_time = time.time()
        
        result = RiskAnalysisResult(
            customer_id=customer_id or str(id(customer_data)),
        )
        
        try:
            query = self._build_query(customer_data)
            
            knowledge_items = self.retrieve(query)
            result.knowledge_used = knowledge_items
            result.retrieval_results = knowledge_items
            
            template = template_type or self.config.template_type
            
            generation_result = self.generator.generate_with_knowledge(
                customer_data=customer_data,
                knowledge_items=knowledge_items,
                template_type=template,
            )
            
            result.generation_result = generation_result
            result.raw_response = generation_result.content
            
            if generation_result.success and generation_result.content:
                parsed = self.output_parser.parse(generation_result.content)
                result.parsed_result = parsed
            else:
                result.success = False
                result.error_message = generation_result.error_message or "LLM生成失败"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.latency = time.time() - start_time
        
        self._update_stats(result)
        
        return result
    
    def analyze_without_retrieval(
        self,
        customer_data: Dict[str, Any],
        customer_id: Optional[str] = None,
    ) -> RiskAnalysisResult:
        """
        不使用检索，直接进行风险分析（纯LLM模式）
        
        Args:
            customer_data: 客户数据
            customer_id: 客户ID
            
        Returns:
            RiskAnalysisResult: 风险分析结果
        """
        start_time = time.time()
        
        result = RiskAnalysisResult(
            customer_id=customer_id or str(id(customer_data)),
        )
        
        try:
            generation_result = self.generator.generate_with_knowledge(
                customer_data=customer_data,
                knowledge_items=[],
                template_type=self.config.template_type,
            )
            
            result.generation_result = generation_result
            result.raw_response = generation_result.content
            
            if generation_result.success and generation_result.content:
                parsed = self.output_parser.parse(generation_result.content)
                result.parsed_result = parsed
            else:
                result.success = False
                result.error_message = generation_result.error_message or "LLM生成失败"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.latency = time.time() - start_time
        self._update_stats(result)
        
        return result
    
    def _update_stats(self, result: RiskAnalysisResult):
        """更新统计信息"""
        self._total_analyses += 1
        self._total_latency += result.latency
        
        if result.generation_result:
            self._total_tokens += result.generation_result.total_tokens
            self._total_cost += result.generation_result.cost
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "total_analyses": self._total_analyses,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 4),
            "average_latency": round(
                self._total_latency / self._total_analyses if self._total_analyses > 0 else 0, 2
            ),
            "config": {
                "retrieval_strategy": self.config.retrieval_strategy.value,
                "top_k": self.config.top_k,
                "template_type": self.config.template_type,
            },
            "generator_stats": self.generator.get_stats(),
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self._total_analyses = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
        self.generator.reset_stats()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_retriever.get_stats()


def create_rag_system(
    retrieval_strategy: str = "hybrid",
    top_k: int = 5,
    temperature: float = 0.3,
    template_type: str = "default",
    persist_dir: Optional[str] = None,
) -> RAGRiskSystem:
    """
    创建RAG风险分析系统实例
    
    Args:
        retrieval_strategy: 检索策略 ('vector' 或 'hybrid')
        top_k: 检索结果数量
        temperature: 生成温度
        template_type: Prompt模板类型
        persist_dir: 向量库持久化目录
        
    Returns:
        RAGRiskSystem: RAG系统实例
    """
    config = RAGSystemConfig(
        retrieval_strategy=RetrievalStrategy(retrieval_strategy),
        top_k=top_k,
        temperature=temperature,
        template_type=template_type,
    )
    return RAGRiskSystem(config=config, persist_dir=persist_dir)
