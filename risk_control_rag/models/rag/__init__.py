"""
RAG模块

包含以下组件：
- embedding: 豆包Embedding向量化
- retriever: 向量检索
- hybrid_retriever: 混合检索
- prompt_builder: Prompt构建
- generator: LLM生成
- output_parser: 输出解析
- rag_system: 完整RAG系统
- batch_predictor: 批量预测
- rag_model: RAG模型
- llm_only: 纯LLM模型
"""

from models.rag.embedding import (
    DoubaoEmbedding,
    EmbeddingCache,
    EmbeddingResult,
    create_embedding_model
)

from models.rag.retriever import (
    VectorRetriever,
    SearchResult,
    RetrievalConfig,
    create_retriever
)

from models.rag.hybrid_retriever import (
    HybridRetriever,
    HybridResult,
    HybridConfig,
    BM25,
    RRFFusion,
    WeightedFusion,
    create_hybrid_retriever
)

from models.rag.prompt_builder import (
    PromptBuilder,
    PromptConfig,
    RetrievedKnowledge,
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
    RiskFactor,
    FavorableFactor,
    parse_risk_analysis,
)

from models.rag.rag_system import (
    RAGRiskSystem,
    RAGSystemConfig,
    RiskAnalysisResult,
    RetrievalStrategy,
    create_rag_system,
)

from models.rag.batch_predictor import (
    BatchPredictor,
    BatchPredictorConfig,
    BatchPredictionResult,
    create_batch_predictor,
)

__all__ = [
    "DoubaoEmbedding",
    "EmbeddingCache",
    "EmbeddingResult",
    "create_embedding_model",
    "VectorRetriever",
    "SearchResult",
    "RetrievalConfig",
    "create_retriever",
    "HybridRetriever",
    "HybridResult",
    "HybridConfig",
    "BM25",
    "RRFFusion",
    "WeightedFusion",
    "create_hybrid_retriever",
    "PromptBuilder",
    "PromptConfig",
    "RetrievedKnowledge",
    "RAGGenerator",
    "GenerationConfig",
    "GenerationResult",
    "create_generator",
    "OutputParser",
    "ParsedRiskAnalysis",
    "RiskLevel",
    "ApprovalSuggestion",
    "RiskFactor",
    "FavorableFactor",
    "parse_risk_analysis",
    "RAGRiskSystem",
    "RAGSystemConfig",
    "RiskAnalysisResult",
    "RetrievalStrategy",
    "create_rag_system",
    "BatchPredictor",
    "BatchPredictorConfig",
    "BatchPredictionResult",
    "create_batch_predictor",
]
