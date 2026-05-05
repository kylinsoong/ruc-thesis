import sys
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.volcengine_api import VolcEngineAPI, TokenUsage, CostEstimate
from config.prompts import (
    RISK_ANALYSIS_SYSTEM_PROMPT,
    get_risk_analysis_prompt,
    get_batch_risk_analysis_prompt,
    parse_risk_level,
    extract_risk_score,
    extract_approval_suggestion,
)


@dataclass
class RiskAnalysisResult:
    application_id: str
    risk_level: int
    risk_level_text: str
    risk_score: int
    approval_suggestion: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchAnalysisSummary:
    total_applications: int = 0
    low_risk_count: int = 0
    medium_risk_count: int = 0
    high_risk_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    review_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    average_latency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMRiskAnalyzer:
    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ):
        self.api = VolcEngineAPI(max_retries=max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = RISK_ANALYSIS_SYSTEM_PROMPT
    
    def analyze_single(
        self,
        customer_data: Dict[str, Any],
        application_id: Optional[str] = None,
    ) -> RiskAnalysisResult:
        if application_id is None:
            application_id = f"APP_{int(time.time() * 1000)}"
        
        prompt = get_risk_analysis_prompt(customer_data)
        
        start_time = time.time()
        response, token_usage, cost = self.api.call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - start_time
        
        risk_level = parse_risk_level(response)
        risk_level_text = ["低风险", "中风险", "高风险"][risk_level]
        risk_score = extract_risk_score(response)
        approval_suggestion = extract_approval_suggestion(response)
        
        return RiskAnalysisResult(
            application_id=application_id,
            risk_level=risk_level,
            risk_level_text=risk_level_text,
            risk_score=risk_score,
            approval_suggestion=approval_suggestion,
            raw_response=response,
            prompt_tokens=token_usage.prompt_tokens,
            completion_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
            cost=cost.total_cost,
            latency=latency,
        )
    
    def analyze_batch(
        self,
        batch_data: List[Dict[str, Any]],
        application_ids: Optional[List[str]] = None,
        show_progress: bool = True,
        delay_between_calls: float = 0.5,
    ) -> Tuple[List[RiskAnalysisResult], BatchAnalysisSummary]:
        results = []
        summary = BatchAnalysisSummary()
        
        if application_ids is None:
            application_ids = [f"APP_{i:04d}" for i in range(len(batch_data))]
        
        iterator = tqdm(zip(batch_data, application_ids), total=len(batch_data), desc="分析进度") if show_progress else zip(batch_data, application_ids)
        
        for customer_data, app_id in iterator:
            result = self.analyze_single(customer_data, app_id)
            results.append(result)
            
            summary.total_applications += 1
            if result.risk_level == 0:
                summary.low_risk_count += 1
            elif result.risk_level == 1:
                summary.medium_risk_count += 1
            else:
                summary.high_risk_count += 1
            
            if result.approval_suggestion == "批准":
                summary.approved_count += 1
            elif result.approval_suggestion == "拒绝":
                summary.rejected_count += 1
            else:
                summary.review_count += 1
            
            summary.total_prompt_tokens += result.prompt_tokens
            summary.total_completion_tokens += result.completion_tokens
            summary.total_tokens += result.total_tokens
            summary.total_cost += result.cost
            summary.total_latency += result.latency
            
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)
        
        if summary.total_applications > 0:
            summary.average_latency = summary.total_latency / summary.total_applications
        
        return results, summary
    
    def analyze_batch_grouped(
        self,
        batch_data: List[Dict[str, Any]],
        group_size: int = 5,
    ) -> Tuple[List[Dict[str, Any]], BatchAnalysisSummary]:
        results = []
        summary = BatchAnalysisSummary()
        
        for i in range(0, len(batch_data), group_size):
            group = batch_data[i:i + group_size]
            prompt = get_batch_risk_analysis_prompt(group)
            
            start_time = time.time()
            response, token_usage, cost = self.api.call_llm(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens * 2,
            )
            latency = time.time() - start_time
            
            for j, data in enumerate(group):
                result = {
                    "application_id": f"APP_{i + j:04d}",
                    "risk_level": parse_risk_level(response),
                    "risk_score": extract_risk_score(response),
                    "raw_response": response,
                }
                results.append(result)
                
                summary.total_applications += 1
            
            summary.total_prompt_tokens += token_usage.prompt_tokens
            summary.total_completion_tokens += token_usage.completion_tokens
            summary.total_tokens += token_usage.total_tokens
            summary.total_cost += cost.total_cost
            summary.total_latency += latency
        
        if summary.total_applications > 0:
            summary.average_latency = summary.total_latency / (len(batch_data) // group_size + 1)
        
        return results, summary
    
    def get_api_stats(self) -> Dict[str, Any]:
        return self.api.get_stats()
    
    def reset_api_stats(self):
        self.api.reset_stats()


def save_results(
    results: List[RiskAnalysisResult],
    output_path: str,
    format: str = "json"
):
    if format == "json":
        data = [r.to_dict() for r in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif format == "csv":
        import csv
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'application_id', 'risk_level', 'risk_level_text',
                'risk_score', 'approval_suggestion', 'total_tokens', 'cost', 'latency'
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'application_id': r.application_id,
                    'risk_level': r.risk_level,
                    'risk_level_text': r.risk_level_text,
                    'risk_score': r.risk_score,
                    'approval_suggestion': r.approval_suggestion,
                    'total_tokens': r.total_tokens,
                    'cost': r.cost,
                    'latency': r.latency,
                })
    else:
        raise ValueError(f"不支持的格式: {format}")


def print_summary(summary: BatchAnalysisSummary):
    print("\n" + "=" * 60)
    print("批量分析统计报告")
    print("=" * 60)
    print(f"总申请数量: {summary.total_applications}")
    print(f"低风险: {summary.low_risk_count} ({summary.low_risk_count/summary.total_applications*100:.1f}%)")
    print(f"中风险: {summary.medium_risk_count} ({summary.medium_risk_count/summary.total_applications*100:.1f}%)")
    print(f"高风险: {summary.high_risk_count} ({summary.high_risk_count/summary.total_applications*100:.1f}%)")
    print("-" * 60)
    print(f"批准建议: {summary.approved_count}")
    print(f"拒绝建议: {summary.rejected_count}")
    print(f"人工复核: {summary.review_count}")
    print("-" * 60)
    print(f"总Token消耗: {summary.total_tokens:,}")
    print(f"  - Prompt Tokens: {summary.total_prompt_tokens:,}")
    print(f"  - Completion Tokens: {summary.total_completion_tokens:,}")
    print(f"总费用: ¥{summary.total_cost:.4f}")
    print(f"平均延迟: {summary.average_latency:.2f}秒")
    print("=" * 60)


if __name__ == "__main__":
    sample_data = {
        "checking_status": "A11",
        "duration": 12,
        "credit_history": "A32",
        "purpose": "A43",
        "credit_amount": 5000,
        "savings_status": "A61",
        "employment": "A73",
        "installment_commitment": 2,
        "personal_status": "A93",
        "other_parties": "A101",
        "residence_since": 2,
        "property_magnitude": "A121",
        "age": 35,
        "other_payment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "num_dependents": 1,
        "own_telephone": "A192",
        "foreign_worker": "A202"
    }
    
    analyzer = LLMRiskAnalyzer()
    
    print("测试单条分析...")
    result = analyzer.analyze_single(sample_data, "TEST_001")
    print(f"风险等级: {result.risk_level_text}")
    print(f"风险评分: {result.risk_score}")
    print(f"审批建议: {result.approval_suggestion}")
    print(f"Token消耗: {result.total_tokens}")
    print(f"费用: ¥{result.cost:.4f}")
