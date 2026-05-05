"""
批量预测模块 - 支持批量风险分析预测

提供批量预测功能，包括：
- 进度显示
- 断点续传
- 结果缓存
- 批量预测报告生成
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

from models.rag.rag_system import RAGRiskSystem, RiskAnalysisResult, RAGSystemConfig


@dataclass
class BatchPredictorConfig:
    """批量预测配置"""
    cache_dir: str = "./cache/batch_predictions"
    cache_enabled: bool = True
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 10
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    delay_between_requests: float = 0.5
    max_retries: int = 3


@dataclass
class BatchPredictionResult:
    """批量预测结果"""
    total_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    cached_count: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    start_time: str = ""
    end_time: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_count": self.total_count,
                "success_count": self.success_count,
                "failed_count": self.failed_count,
                "cached_count": self.cached_count,
                "success_rate": f"{self.success_count / self.total_count * 100:.2f}%" if self.total_count > 0 else "0%",
            },
            "token_usage": {
                "total_tokens": self.total_tokens,
                "total_cost": f"¥{self.total_cost:.4f}",
            },
            "timing": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_latency": f"{self.total_latency:.2f}s",
                "average_latency": f"{self.total_latency / self.success_count:.2f}s" if self.success_count > 0 else "0s",
            },
            "results": self.results,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class BatchPredictor:
    """
    批量预测器
    
    支持批量风险分析预测，提供：
    - 进度显示
    - 断点续传
    - 结果缓存
    - 报告生成
    """
    
    def __init__(
        self,
        rag_system: RAGRiskSystem,
        config: Optional[BatchPredictorConfig] = None,
    ):
        """
        初始化批量预测器
        
        Args:
            rag_system: RAG风险分析系统实例
            config: 批量预测配置
        """
        self.rag_system = rag_system
        self.config = config or BatchPredictorConfig()
        
        if self.config.cache_enabled:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self._checkpoint_file: Optional[str] = None
        self._checkpoint_data: Dict[str, Any] = {}
    
    def _get_cache_key(self, customer_data: Dict[str, Any]) -> str:
        """生成缓存键"""
        data_str = json.dumps(customer_data, sort_keys=True)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.config.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存加载结果"""
        if not self.config.cache_enabled:
            return None
        
        cache_file = self._get_cache_file_path(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """保存结果到缓存"""
        if not self.config.cache_enabled:
            return
        
        cache_file = self._get_cache_file_path(cache_key)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def _init_checkpoint(self, batch_id: str, total_count: int):
        """初始化断点续传"""
        if not self.config.checkpoint_enabled:
            return
        
        self._checkpoint_file = os.path.join(
            self.config.cache_dir, f"checkpoint_{batch_id}.json"
        )
        
        if os.path.exists(self._checkpoint_file):
            try:
                with open(self._checkpoint_file, 'r', encoding='utf-8') as f:
                    self._checkpoint_data = json.load(f)
            except Exception:
                self._checkpoint_data = {
                    "batch_id": batch_id,
                    "total_count": total_count,
                    "completed_indices": [],
                    "results": [],
                }
        else:
            self._checkpoint_data = {
                "batch_id": batch_id,
                "total_count": total_count,
                "completed_indices": [],
                "results": [],
            }
    
    def _save_checkpoint(self, index: int, result: Dict[str, Any]):
        """保存断点"""
        if not self.config.checkpoint_enabled:
            return
        
        self._checkpoint_data["completed_indices"].append(index)
        self._checkpoint_data["results"].append(result)
        
        if len(self._checkpoint_data["completed_indices"]) % self.config.checkpoint_interval == 0:
            try:
                with open(self._checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self._checkpoint_data, f, ensure_ascii=False)
            except Exception:
                pass
    
    def _get_completed_indices(self) -> set:
        """获取已完成的索引"""
        if not self.config.checkpoint_enabled:
            return set()
        return set(self._checkpoint_data.get("completed_indices", []))
    
    def _update_progress(self, current: int, total: int, message: str = ""):
        """更新进度"""
        if self.config.progress_callback:
            self.config.progress_callback(current, total, message)
        else:
            percentage = current / total * 100 if total > 0 else 0
            print(f"\r进度: {current}/{total} ({percentage:.1f}%) {message}", end="", flush=True)
            if current == total:
                print()
    
    def predict_single(
        self,
        customer_data: Dict[str, Any],
        customer_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        单条预测
        
        Args:
            customer_data: 客户数据
            customer_id: 客户ID
            use_cache: 是否使用缓存
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        cache_key = self._get_cache_key(customer_data)
        
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                cached["from_cache"] = True
                return cached
        
        result = self.rag_system.analyze(
            customer_data=customer_data,
            customer_id=customer_id,
        )
        
        result_dict = result.to_dict()
        result_dict["from_cache"] = False
        
        if use_cache:
            self._save_to_cache(cache_key, result_dict)
        
        return result_dict
    
    def predict_batch(
        self,
        batch_data: List[Dict[str, Any]],
        customer_ids: Optional[List[str]] = None,
        batch_id: Optional[str] = None,
        resume: bool = True,
    ) -> BatchPredictionResult:
        """
        批量预测
        
        Args:
            batch_data: 批量客户数据列表
            customer_ids: 客户ID列表
            batch_id: 批次ID（用于断点续传）
            resume: 是否从断点恢复
            
        Returns:
            BatchPredictionResult: 批量预测结果
        """
        batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        total_count = len(batch_data)
        customer_ids = customer_ids or [str(i) for i in range(total_count)]
        
        self._init_checkpoint(batch_id, total_count)
        
        completed_indices = self._get_completed_indices() if resume else set()
        
        result = BatchPredictionResult(
            total_count=total_count,
            start_time=datetime.now().isoformat(),
        )
        
        start_time = time.time()
        
        for idx, customer_data in enumerate(batch_data):
            if idx in completed_indices:
                result.cached_count += 1
                self._update_progress(idx + 1, total_count, f"跳过已完成 #{idx + 1}")
                continue
            
            self._update_progress(idx + 1, total_count, f"处理中 #{idx + 1}")
            
            try:
                single_result = self.predict_single(
                    customer_data=customer_data,
                    customer_id=customer_ids[idx],
                    use_cache=self.config.cache_enabled,
                )
                
                result.results.append(single_result)
                
                if single_result.get("success", False):
                    result.success_count += 1
                    result.total_tokens += single_result.get("token_usage", {}).get("total_tokens", 0)
                    result.total_cost += single_result.get("cost", 0)
                else:
                    result.failed_count += 1
                
                self._save_checkpoint(idx, single_result)
                
            except Exception as e:
                result.failed_count += 1
                error_result = {
                    "customer_id": customer_ids[idx],
                    "success": False,
                    "error_message": str(e),
                }
                result.results.append(error_result)
                self._save_checkpoint(idx, error_result)
            
            if self.config.delay_between_requests > 0 and idx < total_count - 1:
                time.sleep(self.config.delay_between_requests)
        
        result.total_latency = time.time() - start_time
        result.end_time = datetime.now().isoformat()
        
        if self._checkpoint_file and os.path.exists(self._checkpoint_file):
            try:
                os.remove(self._checkpoint_file)
            except Exception:
                pass
        
        return result
    
    def predict_batch_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> BatchPredictionResult:
        """
        从文件读取数据进行批量预测
        
        Args:
            input_file: 输入文件路径（JSON格式）
            output_file: 输出文件路径
            batch_id: 批次ID
            
        Returns:
            BatchPredictionResult: 批量预测结果
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if "customers" in data:
                batch_data = data["customers"]
                customer_ids = data.get("customer_ids", None)
            else:
                batch_data = [data]
                customer_ids = None
        elif isinstance(data, list):
            batch_data = data
            customer_ids = None
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
        
        result = self.predict_batch(
            batch_data=batch_data,
            customer_ids=customer_ids,
            batch_id=batch_id,
        )
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.to_json())
        
        return result
    
    def generate_report(
        self,
        result: BatchPredictionResult,
        output_file: Optional[str] = None,
    ) -> str:
        """
        生成批量预测报告
        
        Args:
            result: 批量预测结果
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("批量风险分析预测报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        report_lines.append("## 预测概要")
        report_lines.append("-" * 40)
        report_lines.append(f"总样本数: {result.total_count}")
        report_lines.append(f"成功数: {result.success_count}")
        report_lines.append(f"失败数: {result.failed_count}")
        report_lines.append(f"缓存命中: {result.cached_count}")
        success_rate = result.success_count / result.total_count * 100 if result.total_count > 0 else 0
        report_lines.append(f"成功率: {success_rate:.2f}%")
        report_lines.append("")
        
        report_lines.append("## Token消耗与费用")
        report_lines.append("-" * 40)
        report_lines.append(f"总Token数: {result.total_tokens}")
        report_lines.append(f"总费用: ¥{result.total_cost:.4f}")
        avg_tokens = result.total_tokens / result.success_count if result.success_count > 0 else 0
        report_lines.append(f"平均Token数/样本: {avg_tokens:.1f}")
        avg_cost = result.total_cost / result.success_count if result.success_count > 0 else 0
        report_lines.append(f"平均费用/样本: ¥{avg_cost:.4f}")
        report_lines.append("")
        
        report_lines.append("## 时间统计")
        report_lines.append("-" * 40)
        report_lines.append(f"开始时间: {result.start_time}")
        report_lines.append(f"结束时间: {result.end_time}")
        report_lines.append(f"总耗时: {result.total_latency:.2f}秒")
        avg_latency = result.total_latency / result.success_count if result.success_count > 0 else 0
        report_lines.append(f"平均耗时/样本: {avg_latency:.2f}秒")
        report_lines.append("")
        
        risk_level_counts = {"低风险": 0, "中风险": 0, "高风险": 0, "未知": 0}
        approval_counts = {"批准": 0, "拒绝": 0, "人工复核": 0, "待定": 0}
        risk_scores = []
        
        for r in result.results:
            if r.get("success", False):
                risk_level = r.get("risk_level", "未知")
                risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
                
                approval = r.get("approval_suggestion", "待定")
                approval_counts[approval] = approval_counts.get(approval, 0) + 1
                
                score = r.get("risk_score", -1)
                if score >= 0:
                    risk_scores.append(score)
        
        report_lines.append("## 风险等级分布")
        report_lines.append("-" * 40)
        for level, count in risk_level_counts.items():
            pct = count / result.success_count * 100 if result.success_count > 0 else 0
            report_lines.append(f"{level}: {count} ({pct:.1f}%)")
        report_lines.append("")
        
        report_lines.append("## 审批建议分布")
        report_lines.append("-" * 40)
        for suggestion, count in approval_counts.items():
            pct = count / result.success_count * 100 if result.success_count > 0 else 0
            report_lines.append(f"{suggestion}: {count} ({pct:.1f}%)")
        report_lines.append("")
        
        if risk_scores:
            report_lines.append("## 风险评分统计")
            report_lines.append("-" * 40)
            report_lines.append(f"平均分: {sum(risk_scores) / len(risk_scores):.1f}")
            report_lines.append(f"最低分: {min(risk_scores)}")
            report_lines.append(f"最高分: {max(risk_scores)}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        report_lines.append("报告结束")
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def clear_cache(self):
        """清空缓存"""
        if os.path.exists(self.config.cache_dir):
            import shutil
            shutil.rmtree(self.config.cache_dir)
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if not self.config.cache_enabled:
            return {"cache_enabled": False}
        
        cache_files = []
        if os.path.exists(self.config.cache_dir):
            cache_files = [f for f in os.listdir(self.config.cache_dir) if f.endswith('.json')]
        
        total_size = 0
        for f in cache_files:
            file_path = os.path.join(self.config.cache_dir, f)
            total_size += os.path.getsize(file_path)
        
        return {
            "cache_enabled": True,
            "cache_dir": self.config.cache_dir,
            "cache_count": len(cache_files),
            "cache_size": f"{total_size / 1024:.2f} KB",
        }


def create_batch_predictor(
    rag_system: RAGRiskSystem,
    cache_enabled: bool = True,
    checkpoint_enabled: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BatchPredictor:
    """
    创建批量预测器实例
    
    Args:
        rag_system: RAG风险分析系统
        cache_enabled: 是否启用缓存
        checkpoint_enabled: 是否启用断点续传
        progress_callback: 进度回调函数
        
    Returns:
        BatchPredictor: 批量预测器实例
    """
    config = BatchPredictorConfig(
        cache_enabled=cache_enabled,
        checkpoint_enabled=checkpoint_enabled,
        progress_callback=progress_callback,
    )
    return BatchPredictor(rag_system=rag_system, config=config)
