"""
风险分析解释样本生成模块

生成用于可解释性评估的风险分析解释样本，包含不同风险等级的案例。
"""

import sys
import os
import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.evaluation_config import ExplainabilityScore, get_scoring_rubric


@dataclass
class RiskAnalysisSample:
    """风险分析样本"""
    sample_id: str
    customer_data: Dict[str, Any]
    risk_level: str
    risk_score: int
    risk_factors: List[Dict[str, Any]]
    favorable_factors: List[Dict[str, Any]]
    approval_suggestion: str
    credit_limit_suggestion: str
    monitoring_indicators: List[str]
    risk_mitigation_measures: List[str]
    explanation: str
    knowledge_references: List[Dict[str, Any]]
    generation_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


LOW_RISK_TEMPLATES = [
    {
        "risk_factors": [
            {"name": "贷款金额适中", "description": "贷款金额占收入比例合理，还款压力可控", "weight": "低"},
            {"name": "贷款期限较长", "description": "贷款期限较长，月供压力分散", "weight": "低"},
        ],
        "favorable_factors": [
            {"name": "信用记录良好", "description": "客户信用历史清白，无逾期记录", "risk_mitigation": "降低违约风险"},
            {"name": "收入稳定", "description": "工作年限长，收入来源稳定", "risk_mitigation": "保障还款能力"},
            {"name": "资产充足", "description": "名下有房产和车辆等资产", "risk_mitigation": "提供担保保障"},
        ],
        "monitoring": ["还款行为监控", "收入变化跟踪", "信用记录更新"],
        "mitigation": ["定期回访", "设置还款提醒"],
    },
    {
        "risk_factors": [
            {"name": "首次贷款客户", "description": "客户为首次申请贷款，缺乏历史参考", "weight": "低"},
        ],
        "favorable_factors": [
            {"name": "职业前景良好", "description": "从事行业前景广阔，职业发展空间大", "risk_mitigation": "预期收入增长"},
            {"name": "负债率低", "description": "现有负债占收入比例低于30%", "risk_mitigation": "还款能力充足"},
            {"name": "年龄优势", "description": "客户年龄适中，处于职业上升期", "risk_mitigation": "长期客户价值"},
        ],
        "monitoring": ["职业状态跟踪", "负债率监控"],
        "mitigation": ["建立客户档案", "定期信用评估"],
    },
]

MEDIUM_RISK_TEMPLATES = [
    {
        "risk_factors": [
            {"name": "负债率偏高", "description": "现有负债占收入比例超过50%", "weight": "中"},
            {"name": "工作年限较短", "description": "当前工作不满2年，职业稳定性待观察", "weight": "中"},
            {"name": "贷款金额较大", "description": "申请贷款金额接近收入上限", "weight": "中"},
        ],
        "favorable_factors": [
            {"name": "有担保人", "description": "提供有稳定收入的担保人", "risk_mitigation": "分散风险"},
            {"name": "学历较高", "description": "本科及以上学历，就业竞争力强", "risk_mitigation": "提升还款能力预期"},
        ],
        "monitoring": ["负债率动态监控", "就业状态跟踪", "担保人资质复核"],
        "mitigation": ["要求定期提供收入证明", "设置预警阈值", "加强贷后管理"],
    },
    {
        "risk_factors": [
            {"name": "信用记录有瑕疵", "description": "存在1-2次短期逾期记录", "weight": "中"},
            {"name": "收入波动较大", "description": "近半年收入波动幅度超过30%", "weight": "中"},
        ],
        "favorable_factors": [
            {"name": "逾期已还清", "description": "历史逾期款项已全部结清", "risk_mitigation": "降低历史风险"},
            {"name": "有抵押物", "description": "提供房产作为抵押", "risk_mitigation": "提供风险缓释"},
        ],
        "monitoring": ["还款行为监控", "收入稳定性跟踪", "抵押物价值评估"],
        "mitigation": ["适当提高首付比例", "缩短贷款期限", "要求购买保险"],
    },
]

HIGH_RISK_TEMPLATES = [
    {
        "risk_factors": [
            {"name": "多头借贷", "description": "同时在3家以上机构申请贷款", "weight": "高"},
            {"name": "收入负债比过高", "description": "月还款额占收入比例超过70%", "weight": "高"},
            {"name": "信用记录不良", "description": "存在多次逾期和违约记录", "weight": "高"},
            {"name": "无稳定收入来源", "description": "近半年无固定收入证明", "weight": "高"},
        ],
        "favorable_factors": [
            {"name": "有第三方担保", "description": "提供担保公司担保", "risk_mitigation": "部分风险缓释"},
        ],
        "monitoring": ["多头借贷监控", "收入来源核实", "担保公司资质评估"],
        "mitigation": ["要求追加抵押物", "降低贷款额度", "加强贷后检查频率"],
    },
    {
        "risk_factors": [
            {"name": "存在欺诈嫌疑", "description": "提供的材料存在不一致之处", "weight": "高"},
            {"name": "关联风险", "description": "关联企业存在违约记录", "weight": "高"},
            {"name": "行业风险", "description": "所处行业景气度下降", "weight": "中"},
        ],
        "favorable_factors": [],
        "monitoring": ["材料真实性核查", "关联方风险监控", "行业动态跟踪"],
        "mitigation": ["启动尽职调查", "要求补充材料", "必要时拒绝申请"],
    },
]


def generate_customer_data(risk_level: str, sample_id: int) -> Dict[str, Any]:
    """根据风险等级生成客户数据"""
    base_data = {
        "sample_id": f"SAMPLE_{sample_id:04d}",
        "age": random.randint(25, 55),
        "gender": random.choice(["男", "女"]),
    }
    
    if risk_level == "低风险":
        base_data.update({
            "employment_years": random.randint(5, 15),
            "monthly_income": random.randint(15000, 30000),
            "existing_debt": random.randint(0, 50000),
            "credit_history": random.choice(["A31", "A32"]),
            "purpose": random.choice(["A40", "A41", "A42"]),
            "credit_amount": random.randint(50000, 150000),
            "duration": random.randint(12, 36),
            "checking_status": random.choice(["A13", "A14"]),
            "savings_status": random.choice(["A63", "A64", "A65"]),
            "housing": random.choice(["A152", "A151"]),
            "property_magnitude": random.randint(2, 4),
        })
    elif risk_level == "中风险":
        base_data.update({
            "employment_years": random.randint(1, 5),
            "monthly_income": random.randint(8000, 15000),
            "existing_debt": random.randint(50000, 150000),
            "credit_history": random.choice(["A32", "A33"]),
            "purpose": random.choice(["A40", "A43", "A46"]),
            "credit_amount": random.randint(100000, 300000),
            "duration": random.randint(24, 48),
            "checking_status": random.choice(["A12", "A13"]),
            "savings_status": random.choice(["A61", "A62"]),
            "housing": random.choice(["A151", "A153"]),
            "property_magnitude": random.randint(1, 3),
        })
    else:
        base_data.update({
            "employment_years": random.randint(0, 2),
            "monthly_income": random.randint(3000, 8000),
            "existing_debt": random.randint(100000, 300000),
            "credit_history": random.choice(["A33", "A34"]),
            "purpose": random.choice(["A44", "A45", "A49"]),
            "credit_amount": random.randint(150000, 500000),
            "duration": random.randint(36, 60),
            "checking_status": random.choice(["A11", "A12"]),
            "savings_status": random.choice(["A61"]),
            "housing": random.choice(["A151", "A153"]),
            "property_magnitude": random.randint(1, 2),
        })
    
    return base_data


def generate_explanation(
    risk_level: str,
    risk_factors: List[Dict],
    favorable_factors: List[Dict],
    approval_suggestion: str,
    customer_data: Dict[str, Any],
    template_idx: int = 0
) -> str:
    """生成风险分析解释文本"""
    
    explanation_parts = []
    
    explanation_parts.append(f"## 风险分析报告\n")
    explanation_parts.append(f"**客户ID**: {customer_data.get('sample_id', 'N/A')}\n")
    explanation_parts.append(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    explanation_parts.append(f"\n### 一、风险等级判定\n")
    explanation_parts.append(f"根据综合评估，该客户的风险等级为**{risk_level}**。\n")
    
    explanation_parts.append(f"\n### 二、风险因素分析\n")
    if risk_factors:
        for i, factor in enumerate(risk_factors, 1):
            explanation_parts.append(f"\n**{i}. {factor['name']}** (权重: {factor.get('weight', '中')})\n")
            explanation_parts.append(f"   {factor['description']}\n")
            if factor.get('source_reference'):
                explanation_parts.append(f"   *参考来源: [{factor['source_reference']}]*\n")
    else:
        explanation_parts.append("\n暂未识别到明显风险因素。\n")
    
    explanation_parts.append(f"\n### 三、有利因素分析\n")
    if favorable_factors:
        for i, factor in enumerate(favorable_factors, 1):
            explanation_parts.append(f"\n**{i}. {factor['name']}**\n")
            explanation_parts.append(f"   {factor['description']}\n")
            if factor.get('risk_mitigation'):
                explanation_parts.append(f"   *风险缓释作用: {factor['risk_mitigation']}*\n")
    else:
        explanation_parts.append("\n暂未识别到明显有利因素。\n")
    
    explanation_parts.append(f"\n### 四、决策建议\n")
    explanation_parts.append(f"**审批建议**: {approval_suggestion}\n")
    
    if risk_level == "低风险":
        explanation_parts.append(f"\n**授信额度建议**: 可给予较高授信额度，建议额度为申请金额的100%。\n")
    elif risk_level == "中风险":
        explanation_parts.append(f"\n**授信额度建议**: 建议适度控制授信额度，可给予申请金额的70%-80%。\n")
    else:
        explanation_parts.append(f"\n**授信额度建议**: 建议严格控制授信额度或拒绝申请。\n")
    
    explanation_parts.append(f"\n### 五、风险监控指标\n")
    
    explanation_parts.append(f"\n### 六、风险缓释措施\n")
    
    explanation_parts.append(f"\n### 七、知识来源引用\n")
    explanation_parts.append(f"- [1] 风险案例: 类似客户违约案例分析\n")
    explanation_parts.append(f"- [2] 监管法规: 个人贷款管理暂行办法\n")
    explanation_parts.append(f"- [3] 行业知识: 信贷风险评估最佳实践\n")
    
    return "".join(explanation_parts)


def generate_sample(
    sample_id: int,
    risk_level: str,
    template_idx: int = 0
) -> RiskAnalysisSample:
    """生成单个风险分析样本"""
    
    if risk_level == "低风险":
        template = LOW_RISK_TEMPLATES[template_idx % len(LOW_RISK_TEMPLATES)]
        risk_score = random.randint(10, 30)
        approval = "批准"
    elif risk_level == "中风险":
        template = MEDIUM_RISK_TEMPLATES[template_idx % len(MEDIUM_RISK_TEMPLATES)]
        risk_score = random.randint(31, 70)
        approval = "人工复核"
    else:
        template = HIGH_RISK_TEMPLATES[template_idx % len(HIGH_RISK_TEMPLATES)]
        risk_score = random.randint(71, 95)
        approval = "拒绝"
    
    customer_data = generate_customer_data(risk_level, sample_id)
    
    risk_factors = template["risk_factors"].copy()
    favorable_factors = template["favorable_factors"].copy()
    monitoring_indicators = template["monitoring"].copy()
    mitigation_measures = template["mitigation"].copy()
    
    credit_limit = ""
    if risk_level == "低风险":
        credit_limit = f"建议授信额度: {customer_data['credit_amount']}元 (100%申请金额)"
    elif risk_level == "中风险":
        limit = int(customer_data['credit_amount'] * 0.75)
        credit_limit = f"建议授信额度: {limit}元 (75%申请金额)"
    else:
        credit_limit = "不建议授信"
    
    explanation = generate_explanation(
        risk_level=risk_level,
        risk_factors=risk_factors,
        favorable_factors=favorable_factors,
        approval_suggestion=approval,
        customer_data=customer_data,
        template_idx=template_idx,
    )
    
    knowledge_refs = [
        {"source_type": "风险案例", "source_id": f"RC{sample_id % 100 + 1:03d}", "relevance": 0.85},
        {"source_type": "监管法规", "source_id": "REG_001", "relevance": 0.75},
        {"source_type": "行业知识", "source_id": "IND_001", "relevance": 0.70},
    ]
    
    return RiskAnalysisSample(
        sample_id=f"SAMPLE_{sample_id:04d}",
        customer_data=customer_data,
        risk_level=risk_level,
        risk_score=risk_score,
        risk_factors=risk_factors,
        favorable_factors=favorable_factors,
        approval_suggestion=approval,
        credit_limit_suggestion=credit_limit,
        monitoring_indicators=monitoring_indicators,
        risk_mitigation_measures=mitigation_measures,
        explanation=explanation,
        knowledge_references=knowledge_refs,
        generation_timestamp=datetime.now().isoformat(),
    )


def generate_samples(
    num_samples: int = 50,
    output_path: Optional[str] = None
) -> List[RiskAnalysisSample]:
    """批量生成风险分析样本"""
    
    samples = []
    
    low_count = num_samples // 5
    medium_count = num_samples // 5 * 2
    high_count = num_samples - low_count - medium_count
    
    sample_id = 1
    
    for i in range(low_count):
        sample = generate_sample(sample_id, "低风险", i)
        samples.append(sample)
        sample_id += 1
    
    for i in range(medium_count):
        sample = generate_sample(sample_id, "中风险", i)
        samples.append(sample)
        sample_id += 1
    
    for i in range(high_count):
        sample = generate_sample(sample_id, "高风险", i)
        samples.append(sample)
        sample_id += 1
    
    random.shuffle(samples)
    
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        samples_dict = [s.to_dict() for s in samples]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples_dict, f, ensure_ascii=False, indent=2)
        
        print(f"已生成 {len(samples)} 个样本，保存至: {output_path}")
    
    return samples


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    output_path = project_root / "results" / "samples" / "risk_analysis_samples.json"
    
    print("=" * 60)
    print("风险分析解释样本生成")
    print("=" * 60)
    
    samples = generate_samples(num_samples=50, output_path=str(output_path))
    
    print(f"\n样本分布统计:")
    risk_levels = {"低风险": 0, "中风险": 0, "高风险": 0}
    for sample in samples:
        risk_levels[sample.risk_level] += 1
    
    for level, count in risk_levels.items():
        print(f"  {level}: {count} 个样本")
    
    print(f"\n样本示例 (第一个):")
    print(f"  样本ID: {samples[0].sample_id}")
    print(f"  风险等级: {samples[0].risk_level}")
    print(f"  风险评分: {samples[0].risk_score}")
    print(f"  审批建议: {samples[0].approval_suggestion}")
    
    print("\n" + "=" * 60)
    print("样本生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
