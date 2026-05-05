"""
RAG Prompt构建模块

负责将检索到的知识与用户查询融合，构建完整的Prompt。
支持多种Prompt模板、上下文长度控制和知识来源标注。
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from string import Template


@dataclass
class RetrievedKnowledge:
    """检索到的知识条目"""
    content: str
    source_type: str
    source_id: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    """Prompt配置"""
    max_context_length: int = 4000
    max_knowledge_items: int = 5
    include_source_annotation: bool = True
    template_type: str = "default"


RAG_SYSTEM_PROMPT = """你是一位资深的金融风控专家，拥有超过15年的信用风险评估经验。你的专业领域包括：
- 个人信贷风险评估
- 企业信用分析
- 反欺诈检测
- 风险模型开发与验证
- 监管合规分析

你的职责是根据提供的客户信息和相关知识库内容，进行专业、客观的风险分析，并给出明确的风险判断和建议。

分析原则：
1. 客观公正：基于事实和数据进行分析，避免主观偏见
2. 全面细致：综合考虑各类风险因素及其相互影响
3. 知识驱动：充分利用提供的知识库内容支持分析结论
4. 风险导向：重点关注可能导致损失的关键风险点
5. 合规优先：确保分析符合金融监管要求
6. 可操作性：提供具体、可执行的风险管理建议"""


RAG_PROMPT_TEMPLATE = """## 任务说明
请基于以下客户信息和相关知识库内容，进行专业的信贷风险评估分析。

## 客户申请信息

$customer_info

## 相关知识库内容

$knowledge_context

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
- 参考知识来源（如有）

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

### 6. 知识来源引用
请标注分析中引用的知识来源编号。

请确保分析专业、客观、有理有据，并充分利用知识库内容支持你的判断。"""


CONCISE_PROMPT_TEMPLATE = """## 客户信息
$customer_info

## 知识库参考
$knowledge_context

请分析风险等级（低/中/高）、主要风险因素（3-5个）、风险评分（0-100分）和审批建议。"""


DETAILED_PROMPT_TEMPLATE = """# 信贷风险评估分析任务

## 一、客户申请信息详情

$customer_info

## 二、相关知识库检索结果

以下是从知识库中检索到的相关内容，请作为分析参考：

$knowledge_context

## 三、分析输出要求

请按照以下详细结构完成风险评估报告：

### 第一部分：风险等级判定
明确给出风险等级判定结果，并说明判定依据。

### 第二部分：风险因素深度分析
#### 2.1 关键风险因素
列出主要风险因素，每个因素需包含：
1. 因素名称与分类
2. 具体风险描述
3. 风险权重评估
4. 与知识库案例的对比分析
5. 知识来源引用

#### 2.2 有利因素
分析申请人的有利因素及其风险缓释作用。

### 第三部分：综合评估
#### 3.1 风险评分
给出0-100分的风险评分及评分依据。

#### 3.2 与历史案例对比
结合知识库中的历史案例进行对比分析。

### 第四部分：决策建议
#### 4.1 审批建议
给出明确的审批建议（批准/拒绝/人工复核）。

#### 4.2 条件性建议
如需附加条件，请详细说明。

#### 4.3 贷后监控建议
提出贷后风险监控的具体建议。

### 第五部分：知识来源索引
列出分析中引用的所有知识来源。"""


KNOWLEDGE_SOURCE_TEMPLATES = {
    "risk_case": "【案例 {source_id}】{content}",
    "regulation": "【法规 {source_id}】{content}",
    "industry_knowledge": "【知识 {source_id}】{content}",
    "default": "【参考 {source_id}】{content}",
}

SOURCE_TYPE_NAMES = {
    "risk_case": "风险案例",
    "regulation": "监管法规",
    "industry_knowledge": "行业知识",
    "default": "参考资料",
}


class PromptBuilder:
    """RAG Prompt构建器"""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        self.templates = {
            "default": RAG_PROMPT_TEMPLATE,
            "concise": CONCISE_PROMPT_TEMPLATE,
            "detailed": DETAILED_PROMPT_TEMPLATE,
        }
    
    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return content[:max_length - 3] + "..."
    
    def _format_knowledge_item(
        self,
        knowledge: RetrievedKnowledge,
        index: int,
        include_annotation: bool = True
    ) -> str:
        source_type = knowledge.source_type or "default"
        template = KNOWLEDGE_SOURCE_TEMPLATES.get(
            source_type,
            KNOWLEDGE_SOURCE_TEMPLATES["default"]
        )
        
        if include_annotation:
            source_name = SOURCE_TYPE_NAMES.get(source_type, "参考资料")
            header = f"[{index}] {source_name}"
            formatted = f"{header}\n{knowledge.content}"
            if knowledge.relevance_score > 0:
                formatted += f"\n(相关度: {knowledge.relevance_score:.2f})"
        else:
            formatted = knowledge.content
        
        return formatted
    
    def _build_knowledge_context(
        self,
        knowledge_items: List[RetrievedKnowledge],
        max_length: int
    ) -> str:
        if not knowledge_items:
            return "（未检索到相关知识库内容）"
        
        selected_items = knowledge_items[:self.config.max_knowledge_items]
        
        context_parts = []
        total_length = 0
        
        for i, item in enumerate(selected_items, 1):
            formatted = self._format_knowledge_item(
                item, i, self.config.include_source_annotation
            )
            
            if total_length + len(formatted) > max_length:
                remaining = max_length - total_length
                if remaining > 100:
                    formatted = self._truncate_content(formatted, remaining)
                    context_parts.append(formatted)
                break
            
            context_parts.append(formatted)
            total_length += len(formatted) + 1
        
        return "\n\n".join(context_parts)
    
    def _format_customer_info(self, customer_data: Dict[str, Any]) -> str:
        from config.prompts import format_customer_info
        return format_customer_info(customer_data)
    
    def build_prompt(
        self,
        customer_data: Dict[str, Any],
        knowledge_items: List[RetrievedKnowledge],
        template_type: Optional[str] = None
    ) -> str:
        template_type = template_type or self.config.template_type
        template_str = self.templates.get(template_type, self.templates["default"])
        
        customer_info = self._format_customer_info(customer_data)
        
        max_context = self.config.max_context_length - len(customer_info) - 500
        knowledge_context = self._build_knowledge_context(knowledge_items, max_context)
        
        template = Template(template_str)
        return template.substitute(
            customer_info=customer_info,
            knowledge_context=knowledge_context
        )
    
    def build_prompt_with_custom_knowledge(
        self,
        customer_data: Dict[str, Any],
        knowledge_context: str,
        template_type: Optional[str] = None
    ) -> str:
        template_type = template_type or self.config.template_type
        template_str = self.templates.get(template_type, self.templates["default"])
        
        customer_info = self._format_customer_info(customer_data)
        
        template = Template(template_str)
        return template.substitute(
            customer_info=customer_info,
            knowledge_context=knowledge_context
        )
    
    def get_system_prompt(self) -> str:
        return RAG_SYSTEM_PROMPT
    
    def add_custom_template(self, name: str, template: str):
        self.templates[name] = template
    
    def estimate_token_count(self, prompt: str) -> int:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', prompt))
        other_chars = len(prompt) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars / 4)
    
    def get_prompt_stats(
        self,
        customer_data: Dict[str, Any],
        knowledge_items: List[RetrievedKnowledge]
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(customer_data, knowledge_items)
        
        return {
            "prompt_length": len(prompt),
            "estimated_tokens": self.estimate_token_count(prompt),
            "knowledge_items_used": min(len(knowledge_items), self.config.max_knowledge_items),
            "template_type": self.config.template_type,
        }


def convert_search_results_to_knowledge(
    search_results: List[Dict[str, Any]]
) -> List[RetrievedKnowledge]:
    knowledge_items = []
    
    for result in search_results:
        metadata = result.get("metadata", {})
        source_type = metadata.get("type", "default")
        source_id = metadata.get("id", "unknown")
        
        distance = result.get("distance", 0)
        relevance = 1 - distance if distance <= 1 else 0
        
        knowledge = RetrievedKnowledge(
            content=result.get("document", ""),
            source_type=source_type,
            source_id=source_id,
            relevance_score=relevance,
            metadata=metadata
        )
        knowledge_items.append(knowledge)
    
    return knowledge_items


def build_rag_prompt(
    customer_data: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    config: Optional[PromptConfig] = None
) -> str:
    builder = PromptBuilder(config)
    knowledge_items = convert_search_results_to_knowledge(search_results)
    return builder.build_prompt(customer_data, knowledge_items)
