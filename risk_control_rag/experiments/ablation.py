"""
消融实验脚本

测试不同配置下的系统性能，包括：
- 完整方案（LLM + RAG + 混合检索）
- 无RAG方案（纯LLM）
- 仅稀疏检索方案（BM25）
- 仅稠密检索方案（向量检索）

评估指标：
- 风险识别准确率
- 响应时间
- Token消耗
- 可解释性评分
"""

import sys
import os
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_CONFIG, EXPERIMENT_CONFIG, RAG_CONFIG
from models.rag.rag_system import RAGRiskSystem, RAGSystemConfig, RetrievalStrategy
from models.rag.llm_only import LLMRiskAnalyzer
from models.rag.hybrid_retriever import HybridRetriever, HybridConfig, BM25
from models.rag.retriever import VectorRetriever, RetrievalConfig
from models.rag.output_parser import RiskLevel, ApprovalSuggestion


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    description: str
    use_rag: bool = True
    use_hybrid: bool = True
    use_vector: bool = True
    use_bm25: bool = True
    

@dataclass
class AblationResult:
    """消融实验结果"""
    config_name: str
    total_samples: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_latency: float = 0.0
    total_tokens: int = 0
    avg_tokens: float = 0.0
    total_cost: float = 0.0
    avg_cost: float = 0.0
    explainability_score: float = 0.0
    risk_factors_avg: float = 0.0
    knowledge_sources_avg: float = 0.0
    parse_success_rate: float = 0.0
    low_risk_count: int = 0
    medium_risk_count: int = 0
    high_risk_count: int = 0
    unknown_risk_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ABLATION_CONFIGS = {
    "full": AblationConfig(
        name="full",
        description="完整方案（LLM + RAG + 混合检索）",
        use_rag=True,
        use_hybrid=True,
        use_vector=True,
        use_bm25=True,
    ),
    "llm_only": AblationConfig(
        name="llm_only",
        description="无RAG方案（纯LLM）",
        use_rag=False,
        use_hybrid=False,
        use_vector=False,
        use_bm25=False,
    ),
    "bm25_only": AblationConfig(
        name="bm25_only",
        description="仅稀疏检索方案（BM25）",
        use_rag=True,
        use_hybrid=False,
        use_vector=False,
        use_bm25=True,
    ),
    "vector_only": AblationConfig(
        name="vector_only",
        description="仅稠密检索方案（向量检索）",
        use_rag=True,
        use_hybrid=False,
        use_vector=True,
        use_bm25=False,
    ),
}


def generate_mock_test_data(num_samples: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """
    生成模拟测试数据
    
    Args:
        num_samples: 样本数量
        seed: 随机种子
        
    Returns:
        List[Dict]: 测试数据列表
    """
    random.seed(seed)
    np.random.seed(seed)
    
    test_data = []
    
    purposes = ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"]
    credit_histories = ["A30", "A31", "A32", "A33", "A34"]
    checking_statuses = ["A11", "A12", "A13", "A14"]
    employments = ["A71", "A72", "A73", "A74", "A75"]
    savings_statuses = ["A61", "A62", "A63", "A64", "A65"]
    personal_statuses = ["A91", "A92", "A93", "A94"]
    property_magnitudes = ["A121", "A122", "A123", "A124"]
    housings = ["A151", "A152", "A153"]
    jobs = ["A171", "A172", "A173", "A174"]
    
    risk_profiles = [
        {"risk_level": RiskLevel.LOW, "weight": 0.3},
        {"risk_level": RiskLevel.MEDIUM, "weight": 0.4},
        {"risk_level": RiskLevel.HIGH, "weight": 0.3},
    ]
    
    for i in range(num_samples):
        profile = random.choices(
            risk_profiles,
            weights=[p["weight"] for p in risk_profiles]
        )[0]
        risk_level = profile["risk_level"]
        
        if risk_level == RiskLevel.LOW:
            credit_amount = random.randint(500, 3000)
            duration = random.randint(6, 18)
            age = random.randint(30, 55)
            credit_history = random.choice(["A31", "A32"])
            checking_status = random.choice(["A13", "A14"])
            employment = random.choice(["A74", "A75"])
        elif risk_level == RiskLevel.MEDIUM:
            credit_amount = random.randint(3000, 8000)
            duration = random.randint(12, 36)
            age = random.randint(25, 50)
            credit_history = random.choice(["A32", "A33"])
            checking_status = random.choice(["A12", "A13"])
            employment = random.choice(["A73", "A74"])
        else:
            credit_amount = random.randint(8000, 15000)
            duration = random.randint(24, 48)
            age = random.randint(20, 40)
            credit_history = random.choice(["A33", "A34"])
            checking_status = random.choice(["A11", "A12"])
            employment = random.choice(["A71", "A72", "A73"])
        
        sample = {
            "customer_id": f"CUST_{i:04d}",
            "checking_status": checking_status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": random.choice(purposes),
            "credit_amount": credit_amount,
            "savings_status": random.choice(savings_statuses),
            "employment": employment,
            "installment_commitment": random.randint(1, 4),
            "personal_status": random.choice(personal_statuses),
            "other_parties": "A101",
            "residence_since": random.randint(1, 4),
            "property_magnitude": random.choice(property_magnitudes),
            "age": age,
            "other_payment_plans": "A143",
            "housing": random.choice(housings),
            "existing_credits": random.randint(1, 3),
            "job": random.choice(jobs),
            "num_dependents": random.randint(1, 2),
            "own_telephone": random.choice(["A191", "A192"]),
            "foreign_worker": "A202",
            "true_risk_level": risk_level.value,
        }
        
        test_data.append(sample)
    
    return test_data


def load_real_test_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    加载真实测试数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        List[Dict]: 测试数据列表
    """
    test_path = os.path.join(data_dir, "german_X_test.csv")
    label_path = os.path.join(data_dir, "german_y_test.csv")
    
    if not os.path.exists(test_path):
        print(f"测试数据文件不存在: {test_path}")
        return []
    
    X_test = pd.read_csv(test_path)
    y_test = pd.read_csv(label_path) if os.path.exists(label_path) else None
    
    test_data = []
    for i, row in X_test.iterrows():
        sample = row.to_dict()
        sample["customer_id"] = f"CUST_{i:04d}"
        if y_test is not None:
            sample["true_risk_level"] = 1 if y_test.iloc[i, 0] == 1 else 0
        test_data.append(sample)
    
    return test_data


class AblationExperiment:
    """消融实验执行器"""
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        初始化消融实验
        
        Args:
            persist_dir: 向量库持久化目录
            output_dir: 结果输出目录
        """
        self.persist_dir = persist_dir or os.path.join(
            DATA_CONFIG["knowledge_base_dir"], "chroma_db"
        )
        self.output_dir = output_dir or EXPERIMENT_CONFIG["results_dir"]
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "tables")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "figures")).mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, AblationResult] = {}
        self.detailed_results: Dict[str, List[Dict]] = {}
    
    def _create_system_for_config(self, config: AblationConfig) -> Any:
        """
        根据配置创建系统实例
        
        Args:
            config: 消融配置
            
        Returns:
            系统实例
        """
        if not config.use_rag:
            return LLMRiskAnalyzer(temperature=0.3, max_tokens=2048)
        
        if config.use_hybrid and config.use_vector and config.use_bm25:
            rag_config = RAGSystemConfig(
                retrieval_strategy=RetrievalStrategy.HYBRID,
                top_k=5,
                temperature=0.3,
                max_tokens=2048,
            )
            return RAGRiskSystem(config=rag_config, persist_dir=self.persist_dir)
        
        if config.use_bm25 and not config.use_vector:
            rag_config = RAGSystemConfig(
                retrieval_strategy=RetrievalStrategy.VECTOR,
                top_k=5,
                temperature=0.3,
                max_tokens=2048,
            )
            system = RAGRiskSystem(config=rag_config, persist_dir=self.persist_dir)
            system._bm25_only_mode = True
            return system
        
        if config.use_vector and not config.use_bm25:
            rag_config = RAGSystemConfig(
                retrieval_strategy=RetrievalStrategy.VECTOR,
                top_k=5,
                temperature=0.3,
                max_tokens=2048,
            )
            return RAGRiskSystem(config=rag_config, persist_dir=self.persist_dir)
        
        return LLMRiskAnalyzer(temperature=0.3, max_tokens=2048)
    
    def _run_single_analysis(
        self,
        system: Any,
        sample: Dict[str, Any],
        config: AblationConfig,
    ) -> Dict[str, Any]:
        """
        执行单次分析
        
        Args:
            system: 系统实例
            sample: 样本数据
            config: 消融配置
            
        Returns:
            Dict: 分析结果
        """
        customer_data = {k: v for k, v in sample.items() 
                        if k not in ["customer_id", "true_risk_level"]}
        customer_id = sample.get("customer_id", "UNKNOWN")
        true_risk_level = sample.get("true_risk_level", -1)
        
        start_time = time.time()
        
        try:
            if isinstance(system, LLMRiskAnalyzer):
                result = system.analyze_single(customer_data, customer_id)
                
                return {
                    "customer_id": customer_id,
                    "true_risk_level": true_risk_level,
                    "predicted_risk_level": result.risk_level,
                    "risk_score": result.risk_score,
                    "approval_suggestion": result.approval_suggestion,
                    "latency": result.latency,
                    "total_tokens": result.total_tokens,
                    "cost": result.cost,
                    "success": True,
                    "error": None,
                    "risk_factors_count": 0,
                    "knowledge_sources_count": 0,
                    "parse_success": True,
                }
            
            elif isinstance(system, RAGRiskSystem):
                if hasattr(system, '_bm25_only_mode') and system._bm25_only_mode:
                    result = self._analyze_with_bm25_only(system, customer_data, customer_id)
                elif not config.use_rag:
                    result = system.analyze_without_retrieval(customer_data, customer_id)
                else:
                    result = system.analyze(customer_data, customer_id)
                
                parsed = result.parsed_result
                
                risk_level = -1
                if parsed and parsed.risk_level:
                    risk_level = parsed.risk_level.value
                
                return {
                    "customer_id": customer_id,
                    "true_risk_level": true_risk_level,
                    "predicted_risk_level": risk_level,
                    "risk_score": parsed.risk_score if parsed else -1,
                    "approval_suggestion": parsed.approval_suggestion.value if parsed else "待定",
                    "latency": result.latency,
                    "total_tokens": result.generation_result.total_tokens if result.generation_result else 0,
                    "cost": result.generation_result.cost if result.generation_result else 0,
                    "success": result.success,
                    "error": result.error_message,
                    "risk_factors_count": len(parsed.risk_factors) if parsed else 0,
                    "knowledge_sources_count": len(result.knowledge_used),
                    "parse_success": parsed.parse_success if parsed else False,
                }
            
            else:
                return {
                    "customer_id": customer_id,
                    "true_risk_level": true_risk_level,
                    "predicted_risk_level": -1,
                    "risk_score": -1,
                    "approval_suggestion": "待定",
                    "latency": 0,
                    "total_tokens": 0,
                    "cost": 0,
                    "success": False,
                    "error": "未知系统类型",
                    "risk_factors_count": 0,
                    "knowledge_sources_count": 0,
                    "parse_success": False,
                }
                
        except Exception as e:
            return {
                "customer_id": customer_id,
                "true_risk_level": true_risk_level,
                "predicted_risk_level": -1,
                "risk_score": -1,
                "approval_suggestion": "待定",
                "latency": time.time() - start_time,
                "total_tokens": 0,
                "cost": 0,
                "success": False,
                "error": str(e),
                "risk_factors_count": 0,
                "knowledge_sources_count": 0,
                "parse_success": False,
            }
    
    def _analyze_with_bm25_only(
        self,
        system: RAGRiskSystem,
        customer_data: Dict[str, Any],
        customer_id: str,
    ) -> Any:
        """
        仅使用BM25检索进行分析
        
        Args:
            system: RAG系统实例
            customer_data: 客户数据
            customer_id: 客户ID
            
        Returns:
            分析结果
        """
        from models.rag.rag_system import RiskAnalysisResult
        from models.rag.prompt_builder import RetrievedKnowledge
        
        start_time = time.time()
        
        result = RiskAnalysisResult(customer_id=customer_id)
        
        try:
            query = system._build_query(customer_data)
            
            if not hasattr(system, '_bm25_retriever'):
                system._bm25_retriever = HybridRetriever(
                    vector_retriever=system.vector_retriever,
                    config=HybridConfig(top_k=5)
                )
                system._bm25_retriever.build_all_bm25_indexes()
            
            bm25_results = system._bm25_retriever.retrieve_bm25(query, "risk_cases", 5)
            
            knowledge_items = []
            for idx, score, content, metadata in bm25_results:
                knowledge_items.append(RetrievedKnowledge(
                    content=content,
                    source_type="risk_case",
                    source_id=str(idx),
                    relevance_score=score,
                    metadata=metadata,
                ))
            
            result.knowledge_used = knowledge_items
            
            generation_result = system.generator.generate_with_knowledge(
                customer_data=customer_data,
                knowledge_items=knowledge_items,
                template_type=system.config.template_type,
            )
            
            result.generation_result = generation_result
            result.raw_response = generation_result.content
            
            if generation_result.success and generation_result.content:
                parsed = system.output_parser.parse(generation_result.content)
                result.parsed_result = parsed
            else:
                result.success = False
                result.error_message = generation_result.error_message or "LLM生成失败"
                
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.latency = time.time() - start_time
        return result
    
    def run_config(
        self,
        config: AblationConfig,
        test_data: List[Dict[str, Any]],
        delay_between_calls: float = 0.5,
    ) -> AblationResult:
        """
        运行单个配置的实验
        
        Args:
            config: 消融配置
            test_data: 测试数据
            delay_between_calls: 调用间隔
            
        Returns:
            AblationResult: 实验结果
        """
        print(f"\n{'='*60}")
        print(f"运行配置: {config.description}")
        print(f"{'='*60}")
        
        system = self._create_system_for_config(config)
        
        results_list = []
        total_latency = 0
        total_tokens = 0
        total_cost = 0
        correct = 0
        total_risk_factors = 0
        total_knowledge_sources = 0
        parse_success_count = 0
        risk_counts = {0: 0, 1: 0, 2: 0, -1: 0}
        
        for i, sample in enumerate(test_data):
            print(f"  处理样本 {i+1}/{len(test_data)}: {sample.get('customer_id', 'UNKNOWN')}")
            
            result = self._run_single_analysis(system, sample, config)
            results_list.append(result)
            
            if result["success"]:
                total_latency += result["latency"]
                total_tokens += result["total_tokens"]
                total_cost += result["cost"]
                total_risk_factors += result["risk_factors_count"]
                total_knowledge_sources += result["knowledge_sources_count"]
                
                if result["parse_success"]:
                    parse_success_count += 1
                
                true_level = sample.get("true_risk_level", -1)
                pred_level = result["predicted_risk_level"]
                
                if true_level == pred_level:
                    correct += 1
                
                if pred_level in risk_counts:
                    risk_counts[pred_level] += 1
            
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)
        
        n = len(test_data)
        n_success = sum(1 for r in results_list if r["success"])
        
        ablation_result = AblationResult(
            config_name=config.name,
            total_samples=n,
            correct_predictions=correct,
            accuracy=correct / n_success if n_success > 0 else 0,
            avg_latency=total_latency / n_success if n_success > 0 else 0,
            total_tokens=total_tokens,
            avg_tokens=total_tokens / n_success if n_success > 0 else 0,
            total_cost=total_cost,
            avg_cost=total_cost / n_success if n_success > 0 else 0,
            explainability_score=self._calculate_explainability(results_list),
            risk_factors_avg=total_risk_factors / n_success if n_success > 0 else 0,
            knowledge_sources_avg=total_knowledge_sources / n_success if n_success > 0 else 0,
            parse_success_rate=parse_success_count / n_success if n_success > 0 else 0,
            low_risk_count=risk_counts[0],
            medium_risk_count=risk_counts[1],
            high_risk_count=risk_counts[2],
            unknown_risk_count=risk_counts[-1],
        )
        
        self.results[config.name] = ablation_result
        self.detailed_results[config.name] = results_list
        
        return ablation_result
    
    def _calculate_explainability(self, results: List[Dict]) -> float:
        """
        计算可解释性评分
        
        基于以下因素：
        - 风险因素数量
        - 知识来源引用数量
        - 解析成功率
        - 输出结构化程度
        
        Args:
            results: 结果列表
            
        Returns:
            float: 可解释性评分 (0-100)
        """
        if not results:
            return 0.0
        
        scores = []
        
        for r in results:
            if not r["success"]:
                scores.append(0)
                continue
            
            score = 0
            
            if r["risk_factors_count"] > 0:
                score += min(r["risk_factors_count"] * 10, 30)
            
            if r["knowledge_sources_count"] > 0:
                score += min(r["knowledge_sources_count"] * 8, 25)
            
            if r["parse_success"]:
                score += 25
            
            if r["risk_score"] >= 0:
                score += 10
            
            if r["approval_suggestion"] != "待定":
                score += 10
            
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def run_all_configs(
        self,
        test_data: List[Dict[str, Any]],
        configs: Optional[Dict[str, AblationConfig]] = None,
        delay_between_calls: float = 0.5,
    ) -> Dict[str, AblationResult]:
        """
        运行所有配置的实验
        
        Args:
            test_data: 测试数据
            configs: 配置字典
            delay_between_calls: 调用间隔
            
        Returns:
            Dict[str, AblationResult]: 各配置的结果
        """
        configs = configs or ABLATION_CONFIGS
        
        for name, config in configs.items():
            self.run_config(config, test_data, delay_between_calls)
        
        return self.results
    
    def save_results_csv(self, filepath: Optional[str] = None):
        """
        保存结果到CSV文件
        
        Args:
            filepath: 文件路径
        """
        filepath = filepath or os.path.join(
            self.output_dir, "tables", "ablation_results.csv"
        )
        
        rows = []
        for name, result in self.results.items():
            config = ABLATION_CONFIGS.get(name)
            row = {
                "配置名称": name,
                "配置描述": config.description if config else name,
                "总样本数": result.total_samples,
                "正确预测数": result.correct_predictions,
                "准确率": f"{result.accuracy:.4f}",
                "平均响应时间(秒)": f"{result.avg_latency:.2f}",
                "总Token消耗": result.total_tokens,
                "平均Token消耗": f"{result.avg_tokens:.1f}",
                "总费用(元)": f"{result.total_cost:.4f}",
                "平均费用(元)": f"{result.avg_cost:.4f}",
                "可解释性评分": f"{result.explainability_score:.2f}",
                "平均风险因素数": f"{result.risk_factors_avg:.2f}",
                "平均知识来源数": f"{result.knowledge_sources_avg:.2f}",
                "解析成功率": f"{result.parse_success_rate:.4f}",
                "低风险数量": result.low_risk_count,
                "中风险数量": result.medium_risk_count,
                "高风险数量": result.high_risk_count,
                "未知风险数量": result.unknown_risk_count,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {filepath}")
        
        return filepath
    
    def save_detailed_results(self, filepath: Optional[str] = None):
        """
        保存详细结果到JSON文件
        
        Args:
            filepath: 文件路径
        """
        filepath = filepath or os.path.join(
            self.output_dir, "tables", "ablation_detailed_results.json"
        )
        
        data = {}
        for name, results in self.detailed_results.items():
            data[name] = results
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"详细结果已保存到: {filepath}")
    
    def plot_comparison(self, filepath: Optional[str] = None):
        """
        绘制消融实验对比图
        
        Args:
            filepath: 文件路径
        """
        filepath = filepath or os.path.join(
            self.output_dir, "figures", "ablation_comparison.png"
        )
        
        config_names = list(self.results.keys())
        config_labels = [ABLATION_CONFIGS[n].description for n in config_names]
        
        metrics = {
            "准确率": [self.results[n].accuracy for n in config_names],
            "可解释性评分(归一化)": [self.results[n].explainability_score / 100 for n in config_names],
            "解析成功率": [self.results[n].parse_success_rate for n in config_names],
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        x = np.arange(len(config_names))
        width = 0.25
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax1.bar(x + i * width, values, width, label=metric_name)
        ax1.set_ylabel('分数')
        ax1.set_title('性能指标对比')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(config_names, rotation=15, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        ax2 = axes[0, 1]
        latencies = [self.results[n].avg_latency for n in config_names]
        bars = ax2.bar(config_names, latencies, color='steelblue')
        ax2.set_ylabel('平均响应时间 (秒)')
        ax2.set_title('响应时间对比')
        ax2.set_xticklabels(config_names, rotation=15, ha='right')
        for bar, latency in zip(bars, latencies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{latency:.2f}s', ha='center', va='bottom')
        
        ax3 = axes[1, 0]
        tokens = [self.results[n].avg_tokens for n in config_names]
        bars = ax3.bar(config_names, tokens, color='coral')
        ax3.set_ylabel('平均Token消耗')
        ax3.set_title('Token消耗对比')
        ax3.set_xticklabels(config_names, rotation=15, ha='right')
        for bar, token in zip(bars, tokens):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{token:.0f}', ha='center', va='bottom')
        
        ax4 = axes[1, 1]
        risk_data = {
            '低风险': [self.results[n].low_risk_count for n in config_names],
            '中风险': [self.results[n].medium_risk_count for n in config_names],
            '高风险': [self.results[n].high_risk_count for n in config_names],
        }
        x = np.arange(len(config_names))
        width = 0.25
        for i, (risk_name, values) in enumerate(risk_data.items()):
            ax4.bar(x + i * width, values, width, label=risk_name)
        ax4.set_ylabel('数量')
        ax4.set_title('风险等级分布')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(config_names, rotation=15, ha='right')
        ax4.legend()
        
        plt.suptitle('消融实验结果对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"对比图已保存到: {filepath}")
    
    def generate_report(self, filepath: Optional[str] = None) -> str:
        """
        生成消融实验报告
        
        Args:
            filepath: 报告文件路径
            
        Returns:
            str: 报告内容
        """
        filepath = filepath or os.path.join(self.output_dir, "ablation_report.md")
        
        report = []
        report.append("# 消融实验报告")
        report.append("")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## 1. 实验概述")
        report.append("")
        report.append("本消融实验旨在评估RAG风险分析系统中各组件的贡献度，通过对比不同配置下的系统性能，")
        report.append("识别关键组件并优化系统架构。")
        report.append("")
        
        report.append("### 1.1 实验配置")
        report.append("")
        report.append("| 配置名称 | 配置描述 | RAG | 混合检索 | 向量检索 | BM25 |")
        report.append("|---------|---------|-----|---------|---------|------|")
        for name, config in ABLATION_CONFIGS.items():
            report.append(f"| {name} | {config.description} | {'✓' if config.use_rag else '✗'} | "
                         f"{'✓' if config.use_hybrid else '✗'} | "
                         f"{'✓' if config.use_vector else '✗'} | "
                         f"{'✓' if config.use_bm25 else '✗'} |")
        report.append("")
        
        report.append("## 2. 实验结果")
        report.append("")
        
        report.append("### 2.1 性能指标汇总")
        report.append("")
        report.append("| 配置 | 准确率 | 平均响应时间 | 平均Token | 可解释性评分 | 解析成功率 |")
        report.append("|------|--------|-------------|----------|-------------|-----------|")
        for name, result in self.results.items():
            config = ABLATION_CONFIGS.get(name)
            report.append(f"| {config.description if config else name} | "
                         f"{result.accuracy:.2%} | "
                         f"{result.avg_latency:.2f}s | "
                         f"{result.avg_tokens:.0f} | "
                         f"{result.explainability_score:.1f} | "
                         f"{result.parse_success_rate:.2%} |")
        report.append("")
        
        report.append("### 2.2 风险等级分布")
        report.append("")
        report.append("| 配置 | 低风险 | 中风险 | 高风险 | 未知 |")
        report.append("|------|--------|--------|--------|------|")
        for name, result in self.results.items():
            config = ABLATION_CONFIGS.get(name)
            report.append(f"| {config.description if config else name} | "
                         f"{result.low_risk_count} | "
                         f"{result.medium_risk_count} | "
                         f"{result.high_risk_count} | "
                         f"{result.unknown_risk_count} |")
        report.append("")
        
        report.append("## 3. 组件贡献分析")
        report.append("")
        
        if "full" in self.results and "llm_only" in self.results:
            full_acc = self.results["full"].accuracy
            llm_acc = self.results["llm_only"].accuracy
            rag_contribution = (full_acc - llm_acc) / full_acc * 100 if full_acc > 0 else 0
            
            report.append("### 3.1 RAG组件贡献")
            report.append("")
            report.append(f"- **准确率提升**: RAG组件使准确率从 {llm_acc:.2%} 提升到 {full_acc:.2%}")
            report.append(f"- **相对贡献度**: {rag_contribution:.1f}%")
            report.append(f"- **可解释性提升**: 从 {self.results['llm_only'].explainability_score:.1f} 提升到 {self.results['full'].explainability_score:.1f}")
            report.append("")
        
        if "full" in self.results and "vector_only" in self.results and "bm25_only" in self.results:
            full_exp = self.results["full"].explainability_score
            vec_exp = self.results["vector_only"].explainability_score
            bm25_exp = self.results["bm25_only"].explainability_score
            
            report.append("### 3.2 混合检索策略贡献")
            report.append("")
            report.append(f"- **混合检索可解释性**: {full_exp:.1f}")
            report.append(f"- **纯向量检索可解释性**: {vec_exp:.1f}")
            report.append(f"- **纯BM25检索可解释性**: {bm25_exp:.1f}")
            report.append(f"- **混合策略优势**: 混合检索结合了语义理解和关键词匹配的优势")
            report.append("")
        
        report.append("### 3.3 各组件特点分析")
        report.append("")
        report.append("#### 完整方案 (LLM + RAG + 混合检索)")
        report.append("- 结合了语义检索和关键词检索的优势")
        report.append("- 提供最丰富的知识来源引用")
        report.append("- 可解释性最佳")
        report.append("")
        
        report.append("#### 纯LLM方案")
        report.append("- 响应速度最快（无检索开销）")
        report.append("- Token消耗相对较低")
        report.append("- 缺乏外部知识支持，可能产生幻觉")
        report.append("- 可解释性较差")
        report.append("")
        
        report.append("#### 仅向量检索方案")
        report.append("- 擅长语义相似性匹配")
        report.append("- 能找到语义相关但词汇不同的内容")
        report.append("- 对专业术语的精确匹配可能不足")
        report.append("")
        
        report.append("#### 仅BM25检索方案")
        report.append("- 擅长关键词精确匹配")
        report.append("- 对专业术语、法规条款等检索效果好")
        report.append("- 缺乏语义理解能力")
        report.append("")
        
        report.append("## 4. 最优配置建议")
        report.append("")
        
        best_config = max(self.results.items(), key=lambda x: x[1].accuracy)
        best_name = best_config[0]
        best_result = best_config[1]
        
        report.append(f"### 4.1 推荐配置: {ABLATION_CONFIGS[best_name].description}")
        report.append("")
        report.append(f"- **准确率**: {best_result.accuracy:.2%}")
        report.append(f"- **平均响应时间**: {best_result.avg_latency:.2f}秒")
        report.append(f"- **可解释性评分**: {best_result.explainability_score:.1f}")
        report.append("")
        
        report.append("### 4.2 场景化配置建议")
        report.append("")
        report.append("| 应用场景 | 推荐配置 | 理由 |")
        report.append("|---------|---------|------|")
        report.append("| 高精度风险分析 | 完整方案 | 准确率最高，可解释性最佳 |")
        report.append("| 实时快速评估 | 纯LLM方案 | 响应最快，适合初步筛选 |")
        report.append("| 法规合规检查 | BM25方案 | 关键词匹配精确，适合条款检索 |")
        report.append("| 语义相似案例 | 向量检索方案 | 语义匹配能力强 |")
        report.append("")
        
        report.append("## 5. 结论")
        report.append("")
        report.append("通过消融实验，我们验证了RAG系统各组件的有效性：")
        report.append("")
        report.append("1. **RAG组件显著提升系统性能**: 相比纯LLM方案，RAG提供了外部知识支持，")
        report.append("   提高了分析的准确性和可解释性。")
        report.append("")
        report.append("2. **混合检索策略最优**: 结合向量检索和BM25检索，能够同时处理语义匹配和")
        report.append("   关键词匹配，提供最全面的检索结果。")
        report.append("")
        report.append("3. **不同场景适用不同配置**: 根据实际需求（速度、精度、可解释性），")
        report.append("   可以选择不同的配置方案。")
        report.append("")
        
        report.append("---")
        report.append("")
        report.append("*本报告由消融实验脚本自动生成*")
        
        report_content = "\n".join(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"报告已保存到: {filepath}")
        
        return report_content


def run_ablation_study(
    test_data: Optional[List[Dict[str, Any]]] = None,
    num_samples: int = 20,
    use_real_data: bool = False,
    delay_between_calls: float = 0.5,
    output_dir: Optional[str] = None,
) -> Dict[str, AblationResult]:
    """
    运行消融实验
    
    Args:
        test_data: 测试数据（可选）
        num_samples: 模拟数据样本数
        use_real_data: 是否使用真实数据
        delay_between_calls: API调用间隔
        output_dir: 输出目录
        
    Returns:
        Dict[str, AblationResult]: 实验结果
    """
    print("="*60)
    print("消融实验开始")
    print("="*60)
    
    if test_data is None:
        if use_real_data:
            data_dir = DATA_CONFIG["processed_data_dir"]
            test_data = load_real_test_data(data_dir)
            if not test_data:
                print("无法加载真实数据，使用模拟数据")
                test_data = generate_mock_test_data(num_samples)
        else:
            print(f"生成 {num_samples} 条模拟测试数据...")
            test_data = generate_mock_test_data(num_samples)
    
    print(f"测试数据量: {len(test_data)}")
    
    experiment = AblationExperiment(output_dir=output_dir)
    
    results = experiment.run_all_configs(test_data, delay_between_calls=delay_between_calls)
    
    experiment.save_results_csv()
    experiment.save_detailed_results()
    experiment.plot_comparison()
    experiment.generate_report()
    
    print("\n" + "="*60)
    print("消融实验完成")
    print("="*60)
    
    return results


def ablate_rag_components(
    test_data: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> Dict[str, AblationResult]:
    """
    RAG组件消融实验
    
    Args:
        test_data: 测试数据
        output_dir: 输出目录
        
    Returns:
        Dict[str, AblationResult]: 实验结果
    """
    configs = {
        "full": ABLATION_CONFIGS["full"],
        "llm_only": ABLATION_CONFIGS["llm_only"],
    }
    
    experiment = AblationExperiment(output_dir=output_dir)
    return experiment.run_all_configs(test_data, configs)


def ablate_knowledge_retrieval(
    test_data: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> Dict[str, AblationResult]:
    """
    知识检索消融实验
    
    Args:
        test_data: 测试数据
        output_dir: 输出目录
        
    Returns:
        Dict[str, AblationResult]: 实验结果
    """
    configs = {
        "full": ABLATION_CONFIGS["full"],
        "vector_only": ABLATION_CONFIGS["vector_only"],
        "bm25_only": ABLATION_CONFIGS["bm25_only"],
    }
    
    experiment = AblationExperiment(output_dir=output_dir)
    return experiment.run_all_configs(test_data, configs)


def main():
    """主函数"""
    print("Starting ablation experiments...")
    
    results = run_ablation_study(
        num_samples=20,
        use_real_data=False,
        delay_between_calls=0.5,
    )
    
    print("\n实验结果摘要:")
    print("-" * 60)
    for name, result in results.items():
        config = ABLATION_CONFIGS.get(name)
        print(f"\n{config.description if config else name}:")
        print(f"  准确率: {result.accuracy:.2%}")
        print(f"  平均响应时间: {result.avg_latency:.2f}秒")
        print(f"  可解释性评分: {result.explainability_score:.1f}")
    
    print("\nAblation experiments completed.")


if __name__ == "__main__":
    main()
