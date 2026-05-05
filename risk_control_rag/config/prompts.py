from typing import Dict, Any, List
from string import Template


RISK_ANALYSIS_SYSTEM_PROMPT = """你是一位资深的金融风控专家，拥有超过15年的信用风险评估经验。你的专业领域包括：
- 个人信贷风险评估
- 企业信用分析
- 反欺诈检测
- 风险模型开发与验证
- 监管合规分析

你的职责是根据提供的客户信息，进行专业、客观的风险分析，并给出明确的风险判断和建议。

分析原则：
1. 客观公正：基于事实和数据进行分析，避免主观偏见
2. 全面细致：综合考虑各类风险因素及其相互影响
3. 风险导向：重点关注可能导致损失的关键风险点
4. 合规优先：确保分析符合金融监管要求
5. 可操作性：提供具体、可执行的风险管理建议"""


RISK_ANALYSIS_PROMPT_TEMPLATE = """请对以下信贷申请进行风险评估分析：

## 申请人基本信息

$customer_info

## 分析要求

请按照以下结构输出风险评估报告：

### 1. 风险等级判定
请给出明确的风险等级：
- **低风险**：申请人信用状况良好，违约概率低
- **中风险**：申请人存在一定风险因素，需要关注
- **高风险**：申请人存在明显风险信号，建议谨慎处理

### 2. 关键风险因素分析
请列出3-5个最主要的风险因素，每个因素包括：
- 因素名称
- 风险描述
- 风险权重（高/中/低）

### 3. 有利因素分析
请列出申请人的有利因素，说明其降低风险的作用。

### 4. 综合风险评分
请给出一个0-100分的风险评分（分数越高风险越大），并说明评分依据。

### 5. 风险管理建议
请提供具体的风险管理建议，包括：
- 是否批准贷款的建议
- 如批准，建议的授信额度和期限
- 需要关注的监控指标
- 其他风险缓释措施

请确保分析专业、客观、有理有据。"""


BATCH_RISK_ANALYSIS_PROMPT = """请对以下多个信贷申请进行批量风险评估：

$batch_info

请为每个申请提供：
1. 申请编号
2. 风险等级（低/中/高）
3. 风险评分（0-100）
4. 主要风险因素（最多3个）
5. 审批建议（批准/拒绝/人工复核）

请以结构化的方式输出结果。"""


FEATURE_DESCRIPTION_TEMPLATE = """
| 特征名称 | 取值 | 说明 |
|---------|------|------|
$feature_rows
"""


GERMAN_CREDIT_FEATURE_MAPPING = {
    "checking_status": {
        "A11": "< 0 DM（账户透支）",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / 薪资分配至少1年",
        "A14": "无支票账户"
    },
    "credit_history": {
        "A30": "没有贷款记录/所有贷款已还清",
        "A31": "所有贷款已按时还清",
        "A32": "现有贷款按时还款至今",
        "A33": "存在延迟还款记录",
        "A34": "存在关键账户/其他银行存在风险"
    },
    "purpose": {
        "A40": "购买新车",
        "A41": "购买二手车",
        "A42": "购买家具/设备",
        "A43": "购买收音机/电视",
        "A44": "购买家用电器",
        "A45": "维修",
        "A46": "教育",
        "A47": "度假",
        "A48": "再培训",
        "A49": "商业用途",
        "A410": "其他"
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "无储蓄账户/未知"
    },
    "employment": {
        "A71": "失业",
        "A72": "就业 < 1年",
        "A73": "就业 1 <= ... < 4年",
        "A75": "就业 >= 7年",
        "A74": "就业 4 <= ... < 7年"
    },
    "personal_status": {
        "A91": "男性 - 离婚/分居",
        "A92": "女性 - 离婚/分居/已婚",
        "A93": "男性 - 单身",
        "A94": "男性 - 已婚/丧偶",
        "A95": "女性 - 单身"
    },
    "other_parties": {
        "A101": "无",
        "A102": "共同申请人",
        "A103": "担保人"
    },
    "property_magnitude": {
        "A121": "房地产",
        "A122": "建筑协会储蓄协议/人寿保险",
        "A123": "汽车/其他",
        "A124": "无财产/未知"
    },
    "other_payment_plans": {
        "A141": "银行",
        "A142": "商店",
        "A143": "无"
    },
    "housing": {
        "A151": "租房",
        "A152": "自有住房",
        "A153": "免费住房"
    },
    "job": {
        "A171": "非技术工人 - 无固定住所",
        "A172": "非技术工人 - 有固定住所",
        "A173": "技术工人/职员",
        "A174": "管理层/个体经营者/高素质员工"
    },
    "own_telephone": {
        "A191": "无",
        "A192": "有（以客户名义注册）"
    },
    "foreign_worker": {
        "A201": "是",
        "A202": "否"
    }
}


FEATURE_NAMES_CN = {
    "checking_status": "支票账户状态",
    "duration": "贷款期限（月）",
    "credit_history": "信用历史",
    "purpose": "贷款目的",
    "credit_amount": "贷款金额（DM）",
    "savings_status": "储蓄账户状态",
    "employment": "就业状态",
    "installment_commitment": "分期付款占收入比例",
    "personal_status": "个人状态",
    "other_parties": "其他担保人",
    "residence_since": "居住时间（年）",
    "property_magnitude": "财产规模",
    "age": "年龄",
    "other_payment_plans": "其他还款计划",
    "housing": "住房类型",
    "existing_credits": "现有信贷数量",
    "job": "工作类型",
    "num_dependents": "受抚养人数",
    "own_telephone": "是否拥有电话",
    "foreign_worker": "是否为外籍工人"
}


def format_customer_info(data: Dict[str, Any]) -> str:
    lines = []
    for key, value in data.items():
        cn_name = FEATURE_NAMES_CN.get(key, key)
        
        if key in GERMAN_CREDIT_FEATURE_MAPPING:
            value_desc = GERMAN_CREDIT_FEATURE_MAPPING[key].get(value, value)
            lines.append(f"- {cn_name}：{value_desc}")
        else:
            lines.append(f"- {cn_name}：{value}")
    
    return "\n".join(lines)


def format_batch_info(batch_data: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, data in enumerate(batch_data, 1):
        lines.append(f"\n### 申请 {idx}")
        lines.append(format_customer_info(data))
    return "\n".join(lines)


def get_risk_analysis_prompt(customer_data: Dict[str, Any]) -> str:
    customer_info = format_customer_info(customer_data)
    template = Template(RISK_ANALYSIS_PROMPT_TEMPLATE)
    return template.substitute(customer_info=customer_info)


def get_batch_risk_analysis_prompt(batch_data: List[Dict[str, Any]]) -> str:
    batch_info = format_batch_info(batch_data)
    template = Template(BATCH_RISK_ANALYSIS_PROMPT)
    return template.substitute(batch_info=batch_info)


RISK_LEVEL_MAPPING = {
    "低风险": 0,
    "中风险": 1,
    "高风险": 2,
    "低": 0,
    "中": 1,
    "高": 2
}


def parse_risk_level(text: str) -> int:
    text_lower = text.lower()
    if "低风险" in text or "低风险" in text_lower:
        return 0
    elif "高风险" in text or "高风险" in text_lower:
        return 2
    elif "中风险" in text or "中风险" in text_lower:
        return 1
    
    for keyword, level in RISK_LEVEL_MAPPING.items():
        if keyword in text:
            return level
    
    return 1


def extract_risk_score(text: str) -> int:
    import re
    patterns = [
        r"风险评分[：:]\s*(\d+)",
        r"评分[：:]\s*(\d+)",
        r"(\d+)\s*分",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 100)
    
    return -1


def extract_approval_suggestion(text: str) -> str:
    text_lower = text.lower()
    if "拒绝" in text or "不批准" in text:
        return "拒绝"
    elif "人工复核" in text or "人工审核" in text:
        return "人工复核"
    elif "批准" in text or "通过" in text:
        return "批准"
    return "待定"
