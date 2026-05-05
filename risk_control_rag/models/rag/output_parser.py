"""
结构化输出解析模块

负责解析LLM生成的风险分析结果，提取风险等级、风险因素、决策建议等结构化信息。
包含完善的容错机制处理解析失败情况。
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    UNKNOWN = -1
    
    @classmethod
    def from_text(cls, text: str) -> "RiskLevel":
        text_lower = text.lower()
        if "低风险" in text or "低风险" in text_lower or "低" in text:
            return cls.LOW
        elif "高风险" in text or "高风险" in text_lower or "高" in text:
            return cls.HIGH
        elif "中风险" in text or "中风险" in text_lower or "中" in text:
            return cls.MEDIUM
        return cls.UNKNOWN
    
    def to_chinese(self) -> str:
        mapping = {
            RiskLevel.LOW: "低风险",
            RiskLevel.MEDIUM: "中风险",
            RiskLevel.HIGH: "高风险",
            RiskLevel.UNKNOWN: "未知",
        }
        return mapping.get(self, "未知")


class ApprovalSuggestion(Enum):
    """审批建议枚举"""
    APPROVED = "批准"
    REJECTED = "拒绝"
    REVIEW = "人工复核"
    PENDING = "待定"
    
    @classmethod
    def from_text(cls, text: str) -> "ApprovalSuggestion":
        text_lower = text.lower()
        if "拒绝" in text or "不批准" in text or "不予批准" in text:
            return cls.REJECTED
        elif "人工复核" in text or "人工审核" in text or "需复核" in text:
            return cls.REVIEW
        elif "批准" in text or "通过" in text or "同意" in text:
            return cls.APPROVED
        return cls.PENDING


@dataclass
class RiskFactor:
    """风险因素"""
    name: str
    description: str
    weight: str = "中"
    source_reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FavorableFactor:
    """有利因素"""
    name: str
    description: str
    risk_mitigation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeSource:
    """知识来源引用"""
    source_type: str
    source_id: str
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedRiskAnalysis:
    """解析后的风险分析结果"""
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    risk_level_text: str = "未知"
    risk_score: int = -1
    risk_factors: List[RiskFactor] = field(default_factory=list)
    favorable_factors: List[FavorableFactor] = field(default_factory=list)
    approval_suggestion: ApprovalSuggestion = ApprovalSuggestion.PENDING
    approval_suggestion_text: str = "待定"
    credit_limit_suggestion: str = ""
    monitoring_indicators: List[str] = field(default_factory=list)
    risk_mitigation_measures: List[str] = field(default_factory=list)
    knowledge_sources: List[KnowledgeSource] = field(default_factory=list)
    raw_response: str = ""
    parse_success: bool = True
    parse_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "risk_level": self.risk_level.value,
            "risk_level_text": self.risk_level_text,
            "risk_score": self.risk_score,
            "risk_factors": [f.to_dict() for f in self.risk_factors],
            "favorable_factors": [f.to_dict() for f in self.favorable_factors],
            "approval_suggestion": self.approval_suggestion.value,
            "approval_suggestion_text": self.approval_suggestion_text,
            "credit_limit_suggestion": self.credit_limit_suggestion,
            "monitoring_indicators": self.monitoring_indicators,
            "risk_mitigation_measures": self.risk_mitigation_measures,
            "knowledge_sources": [s.to_dict() for s in self.knowledge_sources],
            "parse_success": self.parse_success,
            "parse_errors": self.parse_errors,
        }
        return result
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class OutputParser:
    """结构化输出解析器"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._parse_patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "risk_level": re.compile(
                r"(?:风险等级|风险判定)[：:\s]*[*]*([低中高]+风险?)[*]*",
                re.IGNORECASE
            ),
            "risk_score": re.compile(
                r"(?:风险评分|综合评分|评分)[：:\s]*(\d{1,3})\s*(?:分)?",
                re.IGNORECASE
            ),
            "risk_factor_section": re.compile(
                r"(?:关键)?风险因素[^)]*[:：]\s*([\s\S]*?)(?=(?:有利因素|综合|风险评分|决策建议|知识来源|$))",
                re.IGNORECASE
            ),
            "favorable_section": re.compile(
                r"有利因素[^)]*[:：]\s*([\s\S]*?)(?=(?:综合|风险评分|决策建议|知识来源|$))",
                re.IGNORECASE
            ),
            "approval_section": re.compile(
                r"(?:审批|决策|管理)建议[^)]*[:：]\s*([\s\S]*?)(?=(?:知识来源|$))",
                re.IGNORECASE
            ),
            "knowledge_source": re.compile(
                r"\[(\d+)\]\s*(风险案例|监管法规|行业知识|参考资料)",
                re.IGNORECASE
            ),
            "source_ref": re.compile(
                r"参考来源[：:\s]*\[?(\d+)\]?|引用\[?(\d+)\]?|来源\[?(\d+)\]?",
                re.IGNORECASE
            ),
        }
    
    def parse(self, response: str) -> ParsedRiskAnalysis:
        result = ParsedRiskAnalysis(raw_response=response)
        
        try:
            result.risk_level = self._parse_risk_level(response)
            result.risk_level_text = result.risk_level.to_chinese()
        except Exception as e:
            result.parse_errors.append(f"风险等级解析失败: {str(e)}")
        
        try:
            result.risk_score = self._parse_risk_score(response)
        except Exception as e:
            result.parse_errors.append(f"风险评分解析失败: {str(e)}")
        
        try:
            result.risk_factors = self._parse_risk_factors(response)
        except Exception as e:
            result.parse_errors.append(f"风险因素解析失败: {str(e)}")
        
        try:
            result.favorable_factors = self._parse_favorable_factors(response)
        except Exception as e:
            result.parse_errors.append(f"有利因素解析失败: {str(e)}")
        
        try:
            result.approval_suggestion = self._parse_approval_suggestion(response)
            result.approval_suggestion_text = result.approval_suggestion.value
        except Exception as e:
            result.parse_errors.append(f"审批建议解析失败: {str(e)}")
        
        try:
            suggestions = self._parse_suggestions(response)
            result.credit_limit_suggestion = suggestions.get("credit_limit", "")
            result.monitoring_indicators = suggestions.get("monitoring", [])
            result.risk_mitigation_measures = suggestions.get("mitigation", [])
        except Exception as e:
            result.parse_errors.append(f"建议详情解析失败: {str(e)}")
        
        try:
            result.knowledge_sources = self._parse_knowledge_sources(response)
        except Exception as e:
            result.parse_errors.append(f"知识来源解析失败: {str(e)}")
        
        result.parse_success = len(result.parse_errors) == 0
        
        if self.strict_mode and not result.parse_success:
            raise ValueError(f"严格模式下解析失败: {result.parse_errors}")
        
        return result
    
    def _parse_risk_level(self, response: str) -> RiskLevel:
        pattern = self._parse_patterns["risk_level"]
        match = pattern.search(response)
        
        if match:
            return RiskLevel.from_text(match.group(1))
        
        if "低风险" in response or "风险较低" in response:
            return RiskLevel.LOW
        elif "高风险" in response or "风险较高" in response:
            return RiskLevel.HIGH
        elif "中风险" in response or "风险中等" in response:
            return RiskLevel.MEDIUM
        
        return RiskLevel.UNKNOWN
    
    def _parse_risk_score(self, response: str) -> int:
        pattern = self._parse_patterns["risk_score"]
        match = pattern.search(response)
        
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 100)
        
        numbers = re.findall(r'(\d{1,3})\s*(?:分|/100)', response)
        if numbers:
            for num in numbers:
                score = int(num)
                if 0 <= score <= 100:
                    return score
        
        return -1
    
    def _parse_risk_factors(self, response: str) -> List[RiskFactor]:
        factors = []
        
        pattern = self._parse_patterns["risk_factor_section"]
        match = pattern.search(response)
        
        if not match:
            return factors
        
        section = match.group(1)
        
        factor_blocks = re.split(r'(?:^|\n)\s*(?:\d+\.|[-•*])\s*', section)
        
        for block in factor_blocks:
            block = block.strip()
            if not block or len(block) < 5:
                continue
            
            factor = self._parse_single_risk_factor(block)
            if factor:
                factors.append(factor)
        
        return factors[:10]
    
    def _parse_single_risk_factor(self, text: str) -> Optional[RiskFactor]:
        lines = text.strip().split('\n')
        if not lines:
            return None
        
        name = ""
        description = ""
        weight = "中"
        source_ref = None
        
        first_line = lines[0].strip()
        if ':' in first_line or '：' in first_line:
            parts = re.split(r'[:：]', first_line, 1)
            name = parts[0].strip()
            if len(parts) > 1:
                description = parts[1].strip()
        else:
            name = first_line[:50]
            description = first_line
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            if '权重' in line or '风险等级' in line:
                if '高' in line:
                    weight = "高"
                elif '低' in line:
                    weight = "低"
                else:
                    weight = "中"
            elif '描述' in line or '说明' in line:
                desc_match = re.search(r'[:：]\s*(.+)', line)
                if desc_match:
                    description = desc_match.group(1).strip()
            elif '来源' in line or '参考' in line:
                ref_match = re.search(r'\[(\d+)\]', line)
                if ref_match:
                    source_ref = ref_match.group(1)
            else:
                if description:
                    description += " " + line
                else:
                    description = line
        
        if not name:
            return None
        
        return RiskFactor(
            name=name[:100],
            description=description[:500],
            weight=weight,
            source_reference=source_ref
        )
    
    def _parse_favorable_factors(self, response: str) -> List[FavorableFactor]:
        factors = []
        
        pattern = self._parse_patterns["favorable_section"]
        match = pattern.search(response)
        
        if not match:
            return factors
        
        section = match.group(1)
        
        factor_blocks = re.split(r'(?:^|\n)\s*(?:\d+\.|[-•*])\s*', section)
        
        for block in factor_blocks:
            block = block.strip()
            if not block or len(block) < 5:
                continue
            
            factor = self._parse_single_favorable_factor(block)
            if factor:
                factors.append(factor)
        
        return factors[:10]
    
    def _parse_single_favorable_factor(self, text: str) -> Optional[FavorableFactor]:
        lines = text.strip().split('\n')
        if not lines:
            return None
        
        first_line = lines[0].strip()
        name = first_line[:50]
        description = first_line
        mitigation = ""
        
        if ':' in first_line or '：' in first_line:
            parts = re.split(r'[:：]', first_line, 1)
            name = parts[0].strip()
            if len(parts) > 1:
                description = parts[1].strip()
        
        for line in lines[1:]:
            line = line.strip()
            if '降低' in line or '缓释' in line or '作用' in line:
                mitigation = line
            elif description:
                description += " " + line
        
        return FavorableFactor(
            name=name[:100],
            description=description[:500],
            risk_mitigation=mitigation[:200]
        )
    
    def _parse_approval_suggestion(self, response: str) -> ApprovalSuggestion:
        pattern = self._parse_patterns["approval_section"]
        match = pattern.search(response)
        
        if match:
            section = match.group(1)
            result = ApprovalSuggestion.from_text(section)
            if result != ApprovalSuggestion.PENDING:
                return result
        
        approval_patterns = [
            (r'(?:审批建议|决策建议)[：:\s]*[*]*人工复核[*]*', ApprovalSuggestion.REVIEW),
            (r'(?:审批建议|决策建议)[：:\s]*[*]*批准[*]*', ApprovalSuggestion.APPROVED),
            (r'(?:审批建议|决策建议)[：:\s]*[*]*拒绝[*]*', ApprovalSuggestion.REJECTED),
            (r'建议[：:\s]*人工复核', ApprovalSuggestion.REVIEW),
            (r'建议[：:\s]*批准', ApprovalSuggestion.APPROVED),
            (r'建议[：:\s]*拒绝', ApprovalSuggestion.REJECTED),
        ]
        
        for pattern_str, suggestion in approval_patterns:
            if re.search(pattern_str, response, re.IGNORECASE):
                return suggestion
        
        return ApprovalSuggestion.from_text(response)
    
    def _parse_suggestions(self, response: str) -> Dict[str, Any]:
        suggestions = {
            "credit_limit": "",
            "monitoring": [],
            "mitigation": [],
        }
        
        pattern = self._parse_patterns["approval_section"]
        match = pattern.search(response)
        
        if not match:
            return suggestions
        
        section = match.group(1)
        
        credit_match = re.search(
            r'(?:授信额度|额度)[：:\s]*([^\n]+)',
            section
        )
        if credit_match:
            suggestions["credit_limit"] = credit_match.group(1).strip()[:200]
        
        monitor_match = re.search(
            r'(?:监控指标|关注指标)[：:\s]*([\s\S]*?)(?=\n\n|\n[-•*]|\n\d+\.|$)',
            section
        )
        if monitor_match:
            monitor_text = monitor_match.group(1)
            indicators = re.split(r'[,，、\n]', monitor_text)
            suggestions["monitoring"] = [
                ind.strip() for ind in indicators
                if ind.strip() and len(ind.strip()) > 2
            ][:10]
        
        mitigation_match = re.search(
            r'(?:风险缓释|缓释措施|其他措施)[：:\s]*([\s\S]*?)(?=\n\n|\n[-•*]|\n\d+\.|$)',
            section
        )
        if mitigation_match:
            mitigation_text = mitigation_match.group(1)
            measures = re.split(r'[,，、\n]', mitigation_text)
            suggestions["mitigation"] = [
                m.strip() for m in measures
                if m.strip() and len(m.strip()) > 2
            ][:10]
        
        return suggestions
    
    def _parse_knowledge_sources(self, response: str) -> List[KnowledgeSource]:
        sources = []
        
        pattern = self._parse_patterns["knowledge_source"]
        matches = pattern.finditer(response)
        
        seen_ids = set()
        for match in matches:
            source_id = match.group(1)
            source_type = match.group(2)
            
            if source_id not in seen_ids:
                sources.append(KnowledgeSource(
                    source_type=source_type,
                    source_id=source_id,
                ))
                seen_ids.add(source_id)
        
        return sources
    
    def parse_with_fallback(
        self,
        response: str,
        fallback_values: Optional[Dict[str, Any]] = None
    ) -> ParsedRiskAnalysis:
        result = self.parse(response)
        
        if not result.parse_success and fallback_values:
            if result.risk_level == RiskLevel.UNKNOWN and "risk_level" in fallback_values:
                result.risk_level = fallback_values["risk_level"]
                result.risk_level_text = result.risk_level.to_chinese()
            
            if result.risk_score < 0 and "risk_score" in fallback_values:
                result.risk_score = fallback_values["risk_score"]
            
            if result.approval_suggestion == ApprovalSuggestion.PENDING and "approval" in fallback_values:
                result.approval_suggestion = fallback_values["approval"]
                result.approval_suggestion_text = result.approval_suggestion.value
        
        return result
    
    def validate_result(self, result: ParsedRiskAnalysis) -> Tuple[bool, List[str]]:
        errors = []
        
        if result.risk_level == RiskLevel.UNKNOWN:
            errors.append("风险等级未识别")
        
        if result.risk_score < 0:
            errors.append("风险评分未识别")
        
        if not result.risk_factors:
            errors.append("未识别到风险因素")
        
        if result.approval_suggestion == ApprovalSuggestion.PENDING:
            errors.append("审批建议未识别")
        
        risk_score_level = self._score_to_level(result.risk_score)
        if risk_score_level != RiskLevel.UNKNOWN and result.risk_level != RiskLevel.UNKNOWN:
            if risk_score_level != result.risk_level:
                errors.append(
                    f"风险评分({result.risk_score})与风险等级({result.risk_level_text})不一致"
                )
        
        return len(errors) == 0, errors
    
    def _score_to_level(self, score: int) -> RiskLevel:
        if score < 0:
            return RiskLevel.UNKNOWN
        elif score <= 30:
            return RiskLevel.LOW
        elif score <= 70:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH


def parse_risk_analysis(response: str, strict: bool = False) -> ParsedRiskAnalysis:
    parser = OutputParser(strict_mode=strict)
    return parser.parse(response)


def quick_parse_risk_level(response: str) -> Tuple[RiskLevel, int, ApprovalSuggestion]:
    parser = OutputParser()
    result = parser.parse(response)
    return result.risk_level, result.risk_score, result.approval_suggestion
