import os
import time
import json
import tiktoken
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from volcengine.maas import MaasService, MaasException
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CostEstimate:
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "CNY"


@dataclass
class APIStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_cost: CostEstimate = field(default_factory=CostEstimate)
    total_latency: float = 0.0


PRICING = {
    "doubao-seed-2-0-pro-260215": {
        "input": 0.0008,
        "output": 0.002,
        "unit": "per_1k_tokens",
        "currency": "CNY"
    },
    "doubao-embedding-vision-251215": {
        "input": 0.00005,
        "output": 0.0,
        "unit": "per_1k_tokens",
        "currency": "CNY"
    }
}


class VolcEngineAPI:
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ):
        access_key = os.getenv("VOLCENGINE_ACCESS_KEY", "")
        secret_key = os.getenv("VOLCENGINE_SECRET_KEY", "")
        
        if not access_key or not secret_key:
            raise ValueError("VOLCENGINE_ACCESS_KEY 和 VOLCENGINE_SECRET_KEY 环境变量必须设置")
        
        self.maas = MaasService(access_key, secret_key)
        self.endpoint_id = os.getenv("VOLCENGINE_ENDPOINT_ID", "")
        
        if not self.endpoint_id:
            raise ValueError("VOLCENGINE_ENDPOINT_ID 环境变量必须设置")
        
        self.llm_model = "doubao-seed-2-0-pro-260215"
        self.embedding_model = "doubao-embedding-vision-251215"
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        
        self.stats = APIStats()
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text) / 4)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0
    ) -> CostEstimate:
        pricing = PRICING.get(model, {})
        input_price = pricing.get("input", 0) / 1000
        output_price = pricing.get("output", 0) / 1000
        currency = pricing.get("currency", "CNY")
        
        input_cost = input_tokens * input_price
        output_cost = output_tokens * output_price
        
        return CostEstimate(
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(input_cost + output_cost, 6),
            currency=currency
        )

    def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs
    ) -> Tuple[Any, Optional[Exception]]:
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                return result, None
            except MaasException as e:
                last_exception = e
                self.stats.failed_calls += 1
                
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= self.retry_backoff
            except Exception as e:
                last_exception = e
                self.stats.failed_calls += 1
                
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= self.retry_backoff
        
        return None, last_exception

    def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
    ) -> Tuple[str, TokenUsage, CostEstimate]:
        self.stats.total_calls += 1
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        prompt_tokens = self.count_tokens(prompt)
        if system_prompt:
            prompt_tokens += self.count_tokens(system_prompt)
        
        def _make_request():
            req = {
                "model": {
                    "name": self.llm_model,
                },
                "messages": messages,
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
            }
            return self.maas.chat(self.endpoint_id, req)
        
        response, error = self._retry_with_backoff(_make_request)
        
        latency = time.time() - start_time
        self.stats.total_latency += latency
        
        if error:
            print(f"LLM API调用失败（重试{self.max_retries}次后）: {error}")
            return "", TokenUsage(), CostEstimate()
        
        self.stats.successful_calls += 1
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        token_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", prompt_tokens),
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        cost = self.estimate_cost(self.llm_model, token_usage.prompt_tokens, token_usage.completion_tokens)
        
        self.stats.total_tokens.prompt_tokens += token_usage.prompt_tokens
        self.stats.total_tokens.completion_tokens += token_usage.completion_tokens
        self.stats.total_tokens.total_tokens += token_usage.total_tokens
        self.stats.total_cost.input_cost += cost.input_cost
        self.stats.total_cost.output_cost += cost.output_cost
        self.stats.total_cost.total_cost += cost.total_cost
        
        return content, token_usage, cost

    def call_embedding(self, text: str) -> Tuple[List[float], TokenUsage, CostEstimate]:
        self.stats.total_calls += 1
        start_time = time.time()
        
        input_tokens = self.count_tokens(text)
        
        def _make_request():
            req = {
                "model": {
                    "name": self.embedding_model,
                },
                "input": [text],
            }
            return self.maas.embedding(self.endpoint_id, req)
        
        response, error = self._retry_with_backoff(_make_request)
        
        latency = time.time() - start_time
        self.stats.total_latency += latency
        
        if error:
            print(f"Embedding API调用失败（重试{self.max_retries}次后）: {error}")
            return [], TokenUsage(), CostEstimate()
        
        self.stats.successful_calls += 1
        
        embeddings = response.get("data", [{}])[0].get("embedding", [])
        
        usage = response.get("usage", {})
        total_tokens = usage.get("total_tokens", input_tokens)
        
        token_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", input_tokens),
            completion_tokens=0,
            total_tokens=total_tokens
        )
        
        cost = self.estimate_cost(self.embedding_model, token_usage.prompt_tokens, 0)
        
        self.stats.total_tokens.prompt_tokens += token_usage.prompt_tokens
        self.stats.total_tokens.total_tokens += token_usage.total_tokens
        self.stats.total_cost.input_cost += cost.input_cost
        self.stats.total_cost.total_cost += cost.total_cost
        
        return embeddings, token_usage, cost

    def batch_embedding(self, texts: List[str]) -> Tuple[List[List[float]], TokenUsage, CostEstimate]:
        all_embeddings = []
        total_usage = TokenUsage()
        total_cost = CostEstimate()
        
        for text in texts:
            embedding, usage, cost = self.call_embedding(text)
            all_embeddings.append(embedding)
            
            total_usage.prompt_tokens += usage.prompt_tokens
            total_usage.completion_tokens += usage.completion_tokens
            total_usage.total_tokens += usage.total_tokens
            
            total_cost.input_cost += cost.input_cost
            total_cost.output_cost += cost.output_cost
            total_cost.total_cost += cost.total_cost
        
        return all_embeddings, total_usage, total_cost

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "success_rate": (
                self.stats.successful_calls / self.stats.total_calls * 100
                if self.stats.total_calls > 0 else 0
            ),
            "total_tokens": {
                "prompt_tokens": self.stats.total_tokens.prompt_tokens,
                "completion_tokens": self.stats.total_tokens.completion_tokens,
                "total_tokens": self.stats.total_tokens.total_tokens,
            },
            "total_cost": {
                "input_cost": self.stats.total_cost.input_cost,
                "output_cost": self.stats.total_cost.output_cost,
                "total_cost": self.stats.total_cost.total_cost,
                "currency": self.stats.total_cost.currency,
            },
            "average_latency": (
                self.stats.total_latency / self.stats.total_calls
                if self.stats.total_calls > 0 else 0
            ),
        }

    def reset_stats(self):
        self.stats = APIStats()

    def test_connection(self) -> Tuple[bool, str]:
        try:
            content, _, _ = self.call_llm(
                "请回复'连接成功'四个字",
                max_tokens=10
            )
            if content:
                return True, f"连接成功，响应: {content}"
            return False, "连接失败：未收到响应"
        except Exception as e:
            return False, f"连接失败: {str(e)}"


def get_llm_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    api = VolcEngineAPI()
    content, _, _ = api.call_llm(prompt, system_prompt, temperature, max_tokens)
    return content


def get_embedding(text: str) -> List[float]:
    api = VolcEngineAPI()
    embedding, _, _ = api.call_embedding(text)
    return embedding


def get_batch_embeddings(texts: List[str]) -> List[List[float]]:
    api = VolcEngineAPI()
    embeddings, _, _ = api.batch_embedding(texts)
    return embeddings
