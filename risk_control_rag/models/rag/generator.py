"""
RAG生成模块

负责调用LLM进行风险分析生成，支持流式输出、重试机制和Token消耗记录。
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple, Callable, Generator
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.volcengine_api import VolcEngineAPI, TokenUsage, CostEstimate
from models.rag.prompt_builder import (
    PromptBuilder,
    PromptConfig,
    RetrievedKnowledge,
    convert_search_results_to_knowledge,
)


@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    max_retries: int = 3
    retry_delay: float = 1.0
    stream: bool = False


@dataclass
class GenerationResult:
    """生成结果"""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0
    success: bool = True
    error_message: str = ""
    knowledge_sources: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseGenerator(ABC):
    """生成器基类"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        pass


class RAGGenerator(BaseGenerator):
    """RAG生成器"""
    
    def __init__(
        self,
        generation_config: Optional[GenerationConfig] = None,
        prompt_config: Optional[PromptConfig] = None
    ):
        self.generation_config = generation_config or GenerationConfig()
        self.prompt_config = prompt_config or PromptConfig()
        self.prompt_builder = PromptBuilder(self.prompt_config)
        self.api = VolcEngineAPI(
            max_retries=self.generation_config.max_retries,
            retry_delay=self.generation_config.retry_delay,
        )
        
        self._total_generations = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        config = config or self.generation_config
        
        start_time = time.time()
        
        try:
            content, token_usage, cost = self.api.call_llm(
                prompt=prompt,
                system_prompt=system_prompt or self.prompt_builder.get_system_prompt(),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )
            
            latency = time.time() - start_time
            
            self._update_stats(token_usage, cost, latency)
            
            return GenerationResult(
                content=content,
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens,
                cost=cost.total_cost,
                latency=latency,
                success=True,
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return GenerationResult(
                content="",
                latency=latency,
                success=False,
                error_message=str(e),
            )
    
    def generate_with_knowledge(
        self,
        customer_data: Dict[str, Any],
        knowledge_items: List[RetrievedKnowledge],
        template_type: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        config = config or self.generation_config
        
        prompt = self.prompt_builder.build_prompt(
            customer_data, knowledge_items, template_type
        )
        
        result = self.generate(
            prompt=prompt,
            system_prompt=self.prompt_builder.get_system_prompt(),
            config=config
        )
        
        result.knowledge_sources = [
            {
                "source_type": k.source_type,
                "source_id": k.source_id,
                "relevance_score": k.relevance_score,
            }
            for k in knowledge_items[:self.prompt_config.max_knowledge_items]
        ]
        
        return result
    
    def generate_with_search_results(
        self,
        customer_data: Dict[str, Any],
        search_results: List[Dict[str, Any]],
        template_type: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        knowledge_items = convert_search_results_to_knowledge(search_results)
        return self.generate_with_knowledge(
            customer_data, knowledge_items, template_type, config
        )
    
    def generate_batch(
        self,
        batch_data: List[Tuple[Dict[str, Any], List[RetrievedKnowledge]]],
        template_type: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        delay_between_calls: float = 0.5
    ) -> List[GenerationResult]:
        results = []
        
        for i, (customer_data, knowledge_items) in enumerate(batch_data):
            result = self.generate_with_knowledge(
                customer_data, knowledge_items, template_type, config
            )
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(batch_data))
            
            if delay_between_calls > 0 and i < len(batch_data) - 1:
                time.sleep(delay_between_calls)
        
        return results
    
    def _update_stats(
        self,
        token_usage: TokenUsage,
        cost: CostEstimate,
        latency: float
    ):
        self._total_generations += 1
        self._total_tokens += token_usage.total_tokens
        self._total_cost += cost.total_cost
        self._total_latency += latency
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_generations": self._total_generations,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "average_latency": (
                self._total_latency / self._total_generations
                if self._total_generations > 0 else 0
            ),
            "api_stats": self.api.get_stats(),
        }
    
    def reset_stats(self):
        self._total_generations = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
        self.api.reset_stats()
    
    def estimate_cost(
        self,
        customer_data: Dict[str, Any],
        knowledge_items: List[RetrievedKnowledge]
    ) -> Dict[str, float]:
        prompt = self.prompt_builder.build_prompt(customer_data, knowledge_items)
        estimated_tokens = self.prompt_builder.estimate_token_count(prompt)
        
        estimated_output_tokens = self.generation_config.max_tokens // 2
        
        cost = self.api.estimate_cost(
            self.api.llm_model,
            estimated_tokens,
            estimated_output_tokens
        )
        
        return {
            "estimated_input_tokens": estimated_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost": cost.total_cost,
            "currency": cost.currency,
        }


class StreamingGenerator(RAGGenerator):
    """支持流式输出的生成器"""
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> Generator[str, None, GenerationResult]:
        config = config or self.generation_config
        
        full_content = ""
        start_time = time.time()
        
        try:
            content, token_usage, cost = self.api.call_llm(
                prompt=prompt,
                system_prompt=system_prompt or self.prompt_builder.get_system_prompt(),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )
            
            chunk_size = 50
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                full_content += chunk
                yield chunk
            
            latency = time.time() - start_time
            self._update_stats(token_usage, cost, latency)
            
            return GenerationResult(
                content=full_content,
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens,
                cost=cost.total_cost,
                latency=latency,
                success=True,
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return GenerationResult(
                content=full_content,
                latency=latency,
                success=False,
                error_message=str(e),
            )


class RetryableGenerator(RAGGenerator):
    """增强重试机制的生成器"""
    
    def __init__(
        self,
        generation_config: Optional[GenerationConfig] = None,
        prompt_config: Optional[PromptConfig] = None,
        max_total_retries: int = 5,
        backoff_factor: float = 2.0,
        max_backoff: float = 30.0
    ):
        super().__init__(generation_config, prompt_config)
        self.max_total_retries = max_total_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
    
    def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        config = config or self.generation_config
        last_error = None
        backoff = self.generation_config.retry_delay
        
        for attempt in range(self.max_total_retries):
            result = self.generate(prompt, system_prompt, config)
            
            if result.success:
                return result
            
            last_error = result.error_message
            
            if attempt < self.max_total_retries - 1:
                time.sleep(min(backoff, self.max_backoff))
                backoff *= self.backoff_factor
        
        return GenerationResult(
            content="",
            success=False,
            error_message=f"重试{self.max_total_retries}次后仍失败: {last_error}",
        )


def create_generator(
    stream: bool = False,
    enhanced_retry: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    prompt_config: Optional[PromptConfig] = None
) -> BaseGenerator:
    if stream:
        return StreamingGenerator(generation_config, prompt_config)
    elif enhanced_retry:
        return RetryableGenerator(generation_config, prompt_config)
    else:
        return RAGGenerator(generation_config, prompt_config)
