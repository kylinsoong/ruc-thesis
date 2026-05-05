"""
可解释性评估脚本

实现自动评估和人工评估模板生成功能，对比不同模型的解释质量。
"""

import sys
import os
import json
import re
import csv
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.evaluation_config import (
    ExplainabilityScore,
    ExplainabilityDimension,
    EvaluationResult,
    RULE_BASED_INDICATORS,
    ALL_CRITERIA,
    get_scoring_rubric,
)


class RuleBasedEvaluator:
    """基于规则的自动评估器"""
    
    def __init__(self):
        self.indicators = RULE_BASED_INDICATORS
    
    def evaluate_logic_clarity(self, explanation: str) -> Tuple[int, str]:
        """评估逻辑清晰度"""
        score = 3
        comments = []
        
        indicators = self.indicators["logic_clarity"]
        positive_count = 0
        negative_count = 0
        
        for keyword in indicators["positive"]:
            if keyword in explanation:
                positive_count += explanation.count(keyword)
        
        for keyword in indicators["negative"]:
            if keyword in explanation:
                negative_count += explanation.count(keyword)
        
        if positive_count >= 5:
            score = min(5, score + 2)
            comments.append("逻辑连接词使用充分")
        elif positive_count >= 3:
            score = min(5, score + 1)
            comments.append("逻辑连接词使用较好")
        
        if negative_count >= 3:
            score = max(1, score - 1)
            comments.append("存在较多不确定性表述")
        
        if re.search(r'首先.*其次.*最后', explanation, re.DOTALL):
            score = min(5, score + 1)
            comments.append("结构化表述清晰")
        
        return score, "；".join(comments) if comments else "逻辑结构基本合理"
    
    def evaluate_evidence_sufficiency(self, explanation: str) -> Tuple[int, str]:
        """评估证据充分性"""
        score = 3
        comments = []
        
        indicators = self.indicators["evidence_sufficiency"]
        evidence_count = 0
        
        for keyword in indicators["positive"]:
            if keyword in explanation:
                evidence_count += explanation.count(keyword)
        
        if evidence_count >= 8:
            score = 5
            comments.append("证据引用非常充分")
        elif evidence_count >= 5:
            score = 4
            comments.append("证据引用较为充分")
        elif evidence_count >= 2:
            score = 3
            comments.append("有一定证据支持")
        else:
            score = 2
            comments.append("证据支持不足")
        
        if re.search(r'\[\d+\]', explanation):
            score = min(5, score + 1)
            comments.append("包含引用标注")
        
        if "数据" in explanation or "统计" in explanation:
            comments.append("包含数据支撑")
        
        return score, "；".join(comments)
    
    def evaluate_understandability(self, explanation: str) -> Tuple[int, str]:
        """评估可理解性"""
        score = 3
        comments = []
        
        indicators = self.indicators["understandability"]
        clarity_count = 0
        
        for keyword in indicators["positive"]:
            if keyword in explanation:
                clarity_count += 1
        
        if clarity_count >= 3:
            score = min(5, score + 1)
            comments.append("包含解释性表述")
        
        sentences = re.split(r'[。！？\n]', explanation)
        avg_length = sum(len(s) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1)
        
        if avg_length < 50:
            score = min(5, score + 1)
            comments.append("句子长度适中，易于阅读")
        elif avg_length > 100:
            score = max(1, score - 1)
            comments.append("部分句子过长，影响可读性")
        
        if re.search(r'[*]{2}.+[*]{2}', explanation):
            comments.append("使用强调标记，重点突出")
        
        if re.search(r'^#+\s', explanation, re.MULTILINE):
            score = min(5, score + 1)
            comments.append("结构清晰，有标题分层")
        
        return score, "；".join(comments) if comments else "表达基本清晰"
    
    def evaluate_completeness(self, explanation: str) -> Tuple[int, str]:
        """评估完整性"""
        score = 3
        comments = []
        
        required_elements = self.indicators["completeness"]["required_elements"]
        found_elements = []
        
        for element in required_elements:
            if element in explanation:
                found_elements.append(element)
        
        coverage = len(found_elements) / len(required_elements)
        
        if coverage >= 0.9:
            score = 5
            comments.append("要素非常完整")
        elif coverage >= 0.7:
            score = 4
            comments.append("要素较为完整")
        elif coverage >= 0.5:
            score = 3
            comments.append("基本要素齐全")
        elif coverage >= 0.3:
            score = 2
            comments.append("部分要素缺失")
        else:
            score = 1
            comments.append("关键要素缺失")
        
        comments.append(f"包含{len(found_elements)}/{len(required_elements)}项要素")
        
        return score, "；".join(comments)
    
    def evaluate_professionalism(self, explanation: str) -> Tuple[int, str]:
        """评估专业性"""
        score = 3
        comments = []
        
        professional_terms = [
            "风险缓释", "风险敞口", "信用评级", "授信额度",
            "贷后管理", "风险预警", "违约概率", "损失率",
            "担保", "抵押", "质押", "保证",
            "合规", "监管", "内控",
        ]
        
        term_count = 0
        for term in professional_terms:
            if term in explanation:
                term_count += 1
        
        if term_count >= 8:
            score = 5
            comments.append("专业术语使用丰富且准确")
        elif term_count >= 5:
            score = 4
            comments.append("专业术语使用较为充分")
        elif term_count >= 3:
            score = 3
            comments.append("使用了基本专业术语")
        else:
            score = 2
            comments.append("专业术语使用不足")
        
        analysis_patterns = [
            r'分析.*得出',
            r'综合.*评估',
            r'基于.*判断',
            r'根据.*建议',
        ]
        
        pattern_count = sum(1 for p in analysis_patterns if re.search(p, explanation))
        if pattern_count >= 2:
            score = min(5, score + 1)
            comments.append("分析框架规范")
        
        return score, "；".join(comments)
    
    def evaluate(self, sample: Dict[str, Any]) -> ExplainabilityScore:
        """综合评估"""
        explanation = sample.get("explanation", "")
        
        logic_score, _ = self.evaluate_logic_clarity(explanation)
        evidence_score, _ = self.evaluate_evidence_sufficiency(explanation)
        understandability_score, _ = self.evaluate_understandability(explanation)
        completeness_score, _ = self.evaluate_completeness(explanation)
        professionalism_score, _ = self.evaluate_professionalism(explanation)
        
        return ExplainabilityScore(
            logic_clarity=logic_score,
            evidence_sufficiency=evidence_score,
            understandability=understandability_score,
            completeness=completeness_score,
            professionalism=professionalism_score,
        )
    
    def evaluate_with_details(self, sample: Dict[str, Any]) -> Tuple[ExplainabilityScore, Dict[str, str]]:
        """评估并返回详细评论"""
        explanation = sample.get("explanation", "")
        
        logic_score, logic_comment = self.evaluate_logic_clarity(explanation)
        evidence_score, evidence_comment = self.evaluate_evidence_sufficiency(explanation)
        understandability_score, understandability_comment = self.evaluate_understandability(explanation)
        completeness_score, completeness_comment = self.evaluate_completeness(explanation)
        professionalism_score, professionalism_comment = self.evaluate_professionalism(explanation)
        
        scores = ExplainabilityScore(
            logic_clarity=logic_score,
            evidence_sufficiency=evidence_score,
            understandability=understandability_score,
            completeness=completeness_score,
            professionalism=professionalism_score,
        )
        
        comments = {
            "逻辑清晰度": logic_comment,
            "证据充分性": evidence_comment,
            "可理解性": understandability_comment,
            "完整性": completeness_comment,
            "专业性": professionalism_comment,
        }
        
        return scores, comments


def evaluate_samples(
    samples: List[Dict[str, Any]],
    model_type: str = "RAG"
) -> List[EvaluationResult]:
    """批量评估样本"""
    evaluator = RuleBasedEvaluator()
    results = []
    
    for sample in samples:
        scores, comments = evaluator.evaluate_with_details(sample)
        
        result = EvaluationResult(
            sample_id=sample.get("sample_id", ""),
            risk_level=sample.get("risk_level", ""),
            model_type=model_type,
            explanation=sample.get("explanation", ""),
            scores=scores,
            evaluator="auto",
            comments=json.dumps(comments, ensure_ascii=False),
        )
        results.append(result)
    
    return results


def generate_comparison_results(
    samples: List[Dict[str, Any]],
    model_types: List[str] = None
) -> Dict[str, List[EvaluationResult]]:
    """生成不同模型的对比评估结果"""
    if model_types is None:
        model_types = ["RAG", "LLM-Only", "Traditional-ML"]
    
    comparison_results = {}
    
    for model_type in model_types:
        results = evaluate_samples(samples, model_type)
        comparison_results[model_type] = results
    
    return comparison_results


def save_scores_to_csv(
    results: List[EvaluationResult],
    output_path: str
):
    """保存评分结果到CSV"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            "样本ID", "风险等级", "模型类型", "评估者",
            "逻辑清晰度", "证据充分性", "可理解性", "完整性", "专业性", "综合得分",
            "备注"
        ])
        
        for result in results:
            scores = result.scores
            writer.writerow([
                result.sample_id,
                result.risk_level,
                result.model_type,
                result.evaluator,
                scores.logic_clarity,
                scores.evidence_sufficiency,
                scores.understandability,
                scores.completeness,
                scores.professionalism,
                round(scores.get_average_score(), 2),
                result.comments,
            ])
    
    print(f"评分结果已保存至: {output_path}")


def generate_questionnaire(
    samples: List[Dict[str, Any]],
    output_path: str,
    num_samples: int = 10
):
    """生成人工评估问卷"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selected_samples = samples[:num_samples]
    
    rubric = get_scoring_rubric()
    
    questionnaire = []
    questionnaire.append("# 风险分析可解释性人工评估问卷\n")
    questionnaire.append(f"\n**评估日期**: {datetime.now().strftime('%Y-%m-%d')}\n")
    questionnaire.append("**评估说明**: 请根据以下评分标准，对每个风险分析解释进行评分。\n")
    
    questionnaire.append("\n## 评分标准\n\n")
    for dim_name, dim_info in rubric["评分维度"].items():
        questionnaire.append(f"### {dim_name}\n")
        questionnaire.append(f"{dim_info['描述']}\n\n")
        questionnaire.append("| 分数 | 描述 |\n")
        questionnaire.append("|------|------|\n")
        for score, desc in dim_info["评分标准"].items():
            questionnaire.append(f"| {score} | {desc} |\n")
        questionnaire.append("\n")
    
    questionnaire.append("---\n\n")
    questionnaire.append("## 评估样本\n\n")
    
    for i, sample in enumerate(selected_samples, 1):
        questionnaire.append(f"### 样本 {i}\n\n")
        questionnaire.append(f"**样本ID**: {sample.get('sample_id', 'N/A')}\n\n")
        questionnaire.append(f"**风险等级**: {sample.get('risk_level', 'N/A')}\n\n")
        questionnaire.append(f"**风险评分**: {sample.get('risk_score', 'N/A')}\n\n")
        
        questionnaire.append("#### 风险分析解释\n\n")
        questionnaire.append(f"```\n{sample.get('explanation', 'N/A')}\n```\n\n")
        
        questionnaire.append("#### 评分表\n\n")
        questionnaire.append("| 评估维度 | 评分 (1-5) | 备注 |\n")
        questionnaire.append("|----------|------------|------|\n")
        questionnaire.append("| 逻辑清晰度 | | |\n")
        questionnaire.append("| 证据充分性 | | |\n")
        questionnaire.append("| 可理解性 | | |\n")
        questionnaire.append("| 完整性 | | |\n")
        questionnaire.append("| 专业性 | | |\n")
        questionnaire.append("| **综合评价** | | |\n")
        questionnaire.append("\n---\n\n")
    
    questionnaire.append("## 评估者信息\n\n")
    questionnaire.append("- **评估者姓名**: \n")
    questionnaire.append("- **评估者背景**: \n")
    questionnaire.append("- **评估日期**: \n")
    questionnaire.append("- **总体评价**: \n\n")
    
    questionnaire.append("---\n\n")
    questionnaire.append("*感谢您的参与！*\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("".join(questionnaire))
    
    print(f"评估问卷已生成: {output_path}")


def calculate_statistics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """计算评估统计信息"""
    if not results:
        return {}
    
    scores_list = [r.scores for r in results]
    
    stats = {
        "样本数量": len(results),
        "逻辑清晰度": {
            "平均分": sum(s.logic_clarity for s in scores_list) / len(scores_list),
            "最高分": max(s.logic_clarity for s in scores_list),
            "最低分": min(s.logic_clarity for s in scores_list),
        },
        "证据充分性": {
            "平均分": sum(s.evidence_sufficiency for s in scores_list) / len(scores_list),
            "最高分": max(s.evidence_sufficiency for s in scores_list),
            "最低分": min(s.evidence_sufficiency for s in scores_list),
        },
        "可理解性": {
            "平均分": sum(s.understandability for s in scores_list) / len(scores_list),
            "最高分": max(s.understandability for s in scores_list),
            "最低分": min(s.understandability for s in scores_list),
        },
        "完整性": {
            "平均分": sum(s.completeness for s in scores_list) / len(scores_list),
            "最高分": max(s.completeness for s in scores_list),
            "最低分": min(s.completeness for s in scores_list),
        },
        "专业性": {
            "平均分": sum(s.professionalism for s in scores_list) / len(scores_list),
            "最高分": max(s.professionalism for s in scores_list),
            "最低分": min(s.professionalism for s in scores_list),
        },
        "综合得分": {
            "平均分": sum(s.get_average_score() for s in scores_list) / len(scores_list),
        },
    }
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("可解释性评估统计")
    print("=" * 60)
    
    print(f"\n样本数量: {stats['样本数量']}")
    
    print("\n各维度评分统计:")
    for dim in ["逻辑清晰度", "证据充分性", "可理解性", "完整性", "专业性", "综合得分"]:
        if dim in stats:
            print(f"\n  {dim}:")
            for key, value in stats[dim].items():
                print(f"    {key}: {value:.2f}" if isinstance(value, float) else f"    {key}: {value}")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    
    samples_path = project_root / "results" / "samples" / "risk_analysis_samples.json"
    
    if samples_path.exists():
        with open(samples_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"已加载 {len(samples)} 个样本")
    else:
        print("样本文件不存在，请先运行 generate_samples.py 生成样本")
        return
    
    print("\n开始自动评估...")
    evaluator = RuleBasedEvaluator()
    results = []
    
    for sample in samples:
        scores, comments = evaluator.evaluate_with_details(sample)
        result = EvaluationResult(
            sample_id=sample.get("sample_id", ""),
            risk_level=sample.get("risk_level", ""),
            model_type="RAG",
            explanation=sample.get("explanation", ""),
            scores=scores,
            evaluator="auto",
            comments=json.dumps(comments, ensure_ascii=False),
        )
        results.append(result)
    
    scores_path = project_root / "results" / "tables" / "explainability_scores.csv"
    save_scores_to_csv(results, str(scores_path))
    
    stats = calculate_statistics(results)
    print_statistics(stats)
    
    questionnaire_path = project_root / "results" / "questionnaire" / "evaluation_form.md"
    generate_questionnaire(samples, str(questionnaire_path), num_samples=10)
    
    print("\n可解释性评估完成!")


if __name__ == "__main__":
    main()
