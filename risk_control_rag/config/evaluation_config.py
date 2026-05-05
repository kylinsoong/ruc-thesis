"""
可解释性评估配置模块

定义风险分析解释的可解释性评分标准，包括：
- 逻辑清晰度
- 证据充分性
- 可理解性
- 完整性
- 专业性
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class ExplainabilityDimension(Enum):
    """可解释性维度枚举"""
    LOGIC_CLARITY = "逻辑清晰度"
    EVIDENCE_SUFFICIENCY = "证据充分性"
    UNDERSTANDABILITY = "可理解性"
    COMPLETENESS = "完整性"
    PROFESSIONALISM = "专业性"


@dataclass
class ScoringCriteria:
    """评分标准"""
    dimension: str
    min_score: int = 1
    max_score: int = 5
    description: str = ""
    level_descriptions: Dict[int, str] = field(default_factory=dict)
    
    def get_level_description(self, score: int) -> str:
        if score in self.level_descriptions:
            return self.level_descriptions[score]
        return ""
    
    def validate_score(self, score: int) -> bool:
        return self.min_score <= score <= self.max_score


LOGIC_CLARITY_CRITERIA = ScoringCriteria(
    dimension="逻辑清晰度",
    description="评估风险分析解释的逻辑结构是否清晰、推理过程是否合理",
    level_descriptions={
        1: "逻辑混乱，推理过程不清晰，结论缺乏依据",
        2: "逻辑基本存在但不够清晰，部分推理跳跃",
        3: "逻辑结构基本清晰，推理过程可以理解",
        4: "逻辑结构清晰，推理过程合理，因果关系明确",
        5: "逻辑结构非常清晰，推理过程严密，因果关系明确且有说服力",
    }
)

EVIDENCE_SUFFICIENCY_CRITERIA = ScoringCriteria(
    dimension="证据充分性",
    description="评估风险分析解释是否提供了充分的证据支持，包括数据、案例、法规引用等",
    level_descriptions={
        1: "缺乏证据支持，仅凭主观判断",
        2: "证据不足，仅有个别数据或案例支持",
        3: "有一定证据支持，但证据的全面性或权威性不足",
        4: "证据较为充分，包括数据、案例和法规引用",
        5: "证据非常充分，数据详实、案例丰富、法规引用准确",
    }
)

UNDERSTANDABILITY_CRITERIA = ScoringCriteria(
    dimension="可理解性",
    description="评估风险分析解释是否易于理解，语言表达是否清晰，是否使用了恰当的术语解释",
    level_descriptions={
        1: "难以理解，术语使用不当，表达晦涩",
        2: "理解困难，存在较多专业术语未解释",
        3: "基本可以理解，部分术语需要进一步解释",
        4: "易于理解，术语使用恰当且有解释",
        5: "非常易于理解，语言表达清晰，术语解释到位，适合不同背景读者",
    }
)

COMPLETENESS_CRITERIA = ScoringCriteria(
    dimension="完整性",
    description="评估风险分析解释是否涵盖了所有必要的要素，包括风险等级、风险因素、有利因素、决策建议等",
    level_descriptions={
        1: "严重缺失关键要素，无法形成完整的风险分析",
        2: "缺少多个关键要素，风险分析不完整",
        3: "基本要素齐全，但部分内容不够详细",
        4: "要素完整，各部分内容较为详细",
        5: "要素完整且详尽，涵盖风险等级、风险因素、有利因素、决策建议、监控指标等全部内容",
    }
)

PROFESSIONALISM_CRITERIA = ScoringCriteria(
    dimension="专业性",
    description="评估风险分析解释是否体现了专业水准，包括专业术语使用、分析方法、风险识别能力等",
    level_descriptions={
        1: "缺乏专业性，分析方法和术语使用不当",
        2: "专业性不足，存在明显的方法或术语错误",
        3: "具有一定专业性，基本符合行业规范",
        4: "专业性较强，分析方法和术语使用规范",
        5: "专业性很强，体现深厚的风险管理专业素养，符合行业最佳实践",
    }
)

ALL_CRITERIA = {
    ExplainabilityDimension.LOGIC_CLARITY: LOGIC_CLARITY_CRITERIA,
    ExplainabilityDimension.EVIDENCE_SUFFICIENCY: EVIDENCE_SUFFICIENCY_CRITERIA,
    ExplainabilityDimension.UNDERSTANDABILITY: UNDERSTANDABILITY_CRITERIA,
    ExplainabilityDimension.COMPLETENESS: COMPLETENESS_CRITERIA,
    ExplainabilityDimension.PROFESSIONALISM: PROFESSIONALISM_CRITERIA,
}


@dataclass
class ExplainabilityScore:
    """可解释性评分"""
    logic_clarity: int = 3
    evidence_sufficiency: int = 3
    understandability: int = 3
    completeness: int = 3
    professionalism: int = 3
    
    def __post_init__(self):
        for dim in ExplainabilityDimension:
            attr_name = self._dimension_to_attr(dim)
            score = getattr(self, attr_name)
            criteria = ALL_CRITERIA[dim]
            if not criteria.validate_score(score):
                raise ValueError(f"{criteria.dimension}评分必须在{criteria.min_score}-{criteria.max_score}之间")
    
    def _dimension_to_attr(self, dim: ExplainabilityDimension) -> str:
        mapping = {
            ExplainabilityDimension.LOGIC_CLARITY: "logic_clarity",
            ExplainabilityDimension.EVIDENCE_SUFFICIENCY: "evidence_sufficiency",
            ExplainabilityDimension.UNDERSTANDABILITY: "understandability",
            ExplainabilityDimension.COMPLETENESS: "completeness",
            ExplainabilityDimension.PROFESSIONALISM: "professionalism",
        }
        return mapping[dim]
    
    def get_average_score(self) -> float:
        return (self.logic_clarity + self.evidence_sufficiency + 
                self.understandability + self.completeness + self.professionalism) / 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "逻辑清晰度": self.logic_clarity,
            "证据充分性": self.evidence_sufficiency,
            "可理解性": self.understandability,
            "完整性": self.completeness,
            "专业性": self.professionalism,
            "综合得分": round(self.get_average_score(), 2),
        }
    
    def get_dimension_scores(self) -> Dict[str, int]:
        return {
            "逻辑清晰度": self.logic_clarity,
            "证据充分性": self.evidence_sufficiency,
            "可理解性": self.understandability,
            "完整性": self.completeness,
            "专业性": self.professionalism,
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    sample_id: str
    risk_level: str
    model_type: str
    explanation: str
    scores: ExplainabilityScore
    evaluator: str = "auto"
    comments: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "risk_level": self.risk_level,
            "model_type": self.model_type,
            "explanation": self.explanation,
            "scores": self.scores.to_dict(),
            "evaluator": self.evaluator,
            "comments": self.comments,
        }


RULE_BASED_INDICATORS = {
    "logic_clarity": {
        "positive": [
            "因此", "综上所述", "基于以上分析", "由此可见", 
            "主要原因", "根本原因", "导致", "使得",
            "首先", "其次", "最后", "一方面", "另一方面",
        ],
        "negative": [
            "可能", "也许", "大概", "似乎", "好像",
        ],
        "weight": 0.3,
    },
    "evidence_sufficiency": {
        "positive": [
            "根据", "参考", "引用", "数据表明", "统计显示",
            "案例", "法规", "规定", "条款", "第条",
            "历史记录", "过往经验", "行业惯例",
        ],
        "negative": [
            "感觉", "认为", "觉得", "猜测",
        ],
        "weight": 0.3,
    },
    "understandability": {
        "positive": [
            "即", "例如", "比如", "具体来说", "换句话说",
            "简单来说", "通俗地说", "可以理解为",
        ],
        "negative": [
            " aforementioned", "thereof", "herein",
        ],
        "weight": 0.2,
    },
    "completeness": {
        "required_elements": [
            "风险等级", "风险因素", "有利因素", "审批建议", 
            "授信额度", "监控指标", "风险缓释",
        ],
        "weight": 0.2,
    },
}


def get_scoring_rubric() -> Dict[str, Any]:
    """获取评分标准说明"""
    return {
        "评分范围": "1-5分，1分最低，5分最高",
        "评分维度": {
            "逻辑清晰度": {
                "描述": LOGIC_CLARITY_CRITERIA.description,
                "评分标准": LOGIC_CLARITY_CRITERIA.level_descriptions,
            },
            "证据充分性": {
                "描述": EVIDENCE_SUFFICIENCY_CRITERIA.description,
                "评分标准": EVIDENCE_SUFFICIENCY_CRITERIA.level_descriptions,
            },
            "可理解性": {
                "描述": UNDERSTANDABILITY_CRITERIA.description,
                "评分标准": UNDERSTANDABILITY_CRITERIA.level_descriptions,
            },
            "完整性": {
                "描述": COMPLETENESS_CRITERIA.description,
                "评分标准": COMPLETENESS_CRITERIA.level_descriptions,
            },
            "专业性": {
                "描述": PROFESSIONALISM_CRITERIA.description,
                "评分标准": PROFESSIONALISM_CRITERIA.level_descriptions,
            },
        },
        "综合得分": "五个维度的平均分",
    }


def print_scoring_rubric():
    """打印评分标准"""
    rubric = get_scoring_rubric()
    print("=" * 60)
    print("可解释性评分标准")
    print("=" * 60)
    print(f"\n{rubric['评分范围']}\n")
    
    for dim_name, dim_info in rubric["评分维度"].items():
        print(f"\n【{dim_name}】")
        print(f"描述: {dim_info['描述']}")
        print("评分标准:")
        for score, desc in dim_info["评分标准"].items():
            print(f"  {score}分: {desc}")
    
    print(f"\n{rubric['综合得分']}")
    print("=" * 60)
