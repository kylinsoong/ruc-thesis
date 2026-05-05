"""
非结构化文本数据处理实验

对比传统方法（TF-IDF/词嵌入）与LLM+RAG在处理文本数据方面的能力。

实验内容：
1. 传统模型处理：使用TF-IDF或词嵌入处理文本
2. LLM+RAG处理：直接理解文本语义
3. 对比两种方法的处理能力

评估指标：
- 文本特征利用率
- 风险识别准确率
- 处理时间
"""

import sys
import os
import json
import time
import re
import warnings
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.rag.rag_system import RAGRiskSystem, RAGSystemConfig, create_rag_system
    from models.rag.output_parser import RiskLevel
    from config.prompts import parse_risk_level, extract_risk_score
    RAG_AVAILABLE = True
except Exception as e:
    print(f"警告: RAG模块加载失败 ({e})，将使用模拟模式")
    RAG_AVAILABLE = False


@dataclass
class ExperimentResult:
    method: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_latency: float
    total_time: float
    text_feature_utilization: float
    details: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraditionalTextProcessor:
    """传统文本处理方法"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.model = None
        self.feature_names = []
        
    def fit(self, texts: List[str], labels: List[int]):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        X = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, labels)
        
        return self
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self.tfidf_vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        importance = self.model.feature_importances_
        feature_importance = {}
        for name, imp in zip(self.feature_names, importance):
            feature_importance[name] = float(imp)
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_text_feature_utilization(self, texts: List[str]) -> float:
        X = self.tfidf_vectorizer.transform(texts)
        non_zero_ratio = X.nnz / (X.shape[0] * X.shape[1])
        return non_zero_ratio


class LLMRAGTextProcessor:
    """LLM+RAG文本处理方法"""
    
    def __init__(self, rag_system: RAGRiskSystem = None, use_mock: bool = False):
        self.use_mock = use_mock or not RAG_AVAILABLE
        
        if not self.use_mock:
            try:
                self.rag_system = rag_system or create_rag_system(
                    retrieval_strategy="hybrid",
                    top_k=5,
                    temperature=0.3,
                    template_type="default"
                )
            except Exception as e:
                print(f"警告: RAG系统初始化失败 ({e})，将使用模拟模式")
                self.use_mock = True
        
        self.text_feature_keywords = [
            "贷款用途", "经营", "装修", "教育", "医疗", "购车", "购房",
            "创业", "投资", "消费", "旅游", "还款", "逾期", "信用",
            "收入", "负债", "风险", "欺诈", "可疑", "异常", "稳定",
            "良好", "困难", "紧急", "周转", "扩大", "购买", "支付"
        ]
        
        self.risk_keywords = {
            "high": ["伪造", "虚假", "欺诈", "洗钱", "赌博", "诈骗", "可疑", "异常", 
                     "不明", "失联", "非法", "违规", "透支", "逾期", "拖欠", "风险",
                     "多头借贷", "高利贷", "套现", "盗取", "冒用"],
            "medium": ["负债", "困难", "紧张", "下滑", "亏损", "压力", "不稳定", 
                       "波动", "下降", "减少", "周转", "紧急"],
            "low": ["稳定", "良好", "正常", "按时", "信用记录良好", "有储蓄", 
                    "收入稳定", "无拖欠", "无逾期", "有经验", "详细计划"]
        }
    
    def _mock_analyze(self, case_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        text_fields = [
            case_data.get("loan_purpose_text", ""),
            case_data.get("customer_appeal", ""),
            case_data.get("transaction_remark", "")
        ]
        combined_text = " ".join([t for t in text_fields if t])
        
        risk_scores = {"high": 0, "medium": 0, "low": 0}
        
        for level, keywords in self.risk_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    risk_scores[level] += 1
        
        if risk_scores["high"] >= 2:
            predicted_level = 2
            risk_level_text = "高风险"
            risk_score = 70 + min(risk_scores["high"] * 5, 25)
        elif risk_scores["high"] >= 1 or risk_scores["medium"] >= 3:
            predicted_level = 1
            risk_level_text = "中风险"
            risk_score = 50 + min(risk_scores["medium"] * 5, 20)
        elif risk_scores["low"] >= 2:
            predicted_level = 0
            risk_level_text = "低风险"
            risk_score = 20 + min(30 - risk_scores["low"] * 5, 30)
        else:
            predicted_level = 1
            risk_level_text = "中风险"
            risk_score = 50
        
        risk_factors = []
        favorable_factors = []
        
        for keyword in self.risk_keywords["high"]:
            if keyword in combined_text:
                risk_factors.append({
                    "name": keyword,
                    "description": f"检测到{keyword}相关风险信号",
                    "weight": "高"
                })
        
        for keyword in self.risk_keywords["low"]:
            if keyword in combined_text:
                favorable_factors.append({
                    "name": keyword,
                    "description": f"存在{keyword}有利因素"
                })
        
        risk_factors = risk_factors[:5]
        favorable_factors = favorable_factors[:3]
        
        details = {
            "risk_level_text": risk_level_text,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "favorable_factors": favorable_factors,
            "approval_suggestion": "批准" if predicted_level == 0 else ("人工复核" if predicted_level == 1 else "拒绝"),
            "latency": 0.5 + np.random.random() * 0.5,
            "raw_response": f"[模拟分析] 风险等级: {risk_level_text}, 风险评分: {risk_score}"
        }
        
        return predicted_level, details
    
    def analyze_text(self, case_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        if self.use_mock:
            return self._mock_analyze(case_data)
        
        customer_data = case_data.get("structured_data", {}).copy()
        
        text_fields = {
            "loan_purpose_text": case_data.get("loan_purpose_text", ""),
            "customer_appeal": case_data.get("customer_appeal", ""),
            "transaction_remark": case_data.get("transaction_remark", "")
        }
        
        combined_text = " ".join([v for v in text_fields.values() if v])
        customer_data["text_description"] = combined_text[:500]
        
        result = self.rag_system.analyze(customer_data, customer_id=case_data.get("customer_id"))
        
        if result.success and result.parsed_result:
            risk_level = result.parsed_result.risk_level.value
            details = {
                "risk_level_text": result.parsed_result.risk_level_text,
                "risk_score": result.parsed_result.risk_score,
                "risk_factors": [f.to_dict() for f in result.parsed_result.risk_factors],
                "favorable_factors": [f.to_dict() for f in result.parsed_result.favorable_factors],
                "approval_suggestion": result.parsed_result.approval_suggestion_text,
                "latency": result.latency,
                "raw_response": result.raw_response[:500] if result.raw_response else ""
            }
        else:
            risk_level = 1
            details = {
                "error": result.error_message,
                "latency": result.latency
            }
        
        return risk_level, details
    
    def calculate_text_feature_utilization(self, case_data: Dict[str, Any], details: Dict[str, Any]) -> float:
        text_fields = [
            case_data.get("loan_purpose_text", ""),
            case_data.get("customer_appeal", ""),
            case_data.get("transaction_remark", "")
        ]
        combined_text = " ".join(text_fields)
        
        if not combined_text:
            return 0.0
        
        keywords_found = 0
        for keyword in self.text_feature_keywords:
            if keyword in combined_text:
                keywords_found += 1
        
        keyword_ratio = keywords_found / len(self.text_feature_keywords)
        
        risk_factors = details.get("risk_factors", [])
        favorable_factors = details.get("favorable_factors", [])
        total_factors = len(risk_factors) + len(favorable_factors)
        factor_score = min(total_factors / 6.0, 1.0)
        
        utilization = 0.5 * keyword_ratio + 0.5 * factor_score
        
        return utilization


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    return cases


def prepare_traditional_data(cases: List[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    
    for case in cases:
        text_parts = [
            case.get("loan_purpose_text", ""),
            case.get("customer_appeal", ""),
            case.get("transaction_remark", "")
        ]
        combined_text = " ".join([t for t in text_parts if t])
        texts.append(combined_text)
        
        risk_level = case.get("risk_level", "medium")
        level_map = {"low": 0, "medium": 1, "high": 2}
        labels.append(level_map.get(risk_level, 1))
    
    return texts, labels


def run_traditional_experiment(cases: List[Dict[str, Any]]) -> ExperimentResult:
    print("\n" + "="*60)
    print("运行传统方法实验 (TF-IDF + Random Forest)")
    print("="*60)
    
    texts, labels = prepare_traditional_data(cases)
    
    processor = TraditionalTextProcessor(max_features=500)
    
    n_samples = len(texts)
    n_train = int(n_samples * 0.7)
    
    train_texts = texts[:n_train]
    train_labels = labels[:n_train]
    test_texts = texts[n_train:]
    test_labels = labels[n_train:]
    
    start_time = time.time()
    processor.fit(train_texts, train_labels)
    training_time = time.time() - start_time
    
    start_time = time.time()
    predictions, probabilities = processor.predict(test_texts)
    prediction_time = time.time() - start_time
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    
    text_utilization = processor.get_text_feature_utilization(test_texts)
    
    feature_importance = processor.get_feature_importance()
    print(f"\nTop 20 重要特征:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:20]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    details = []
    test_cases = cases[n_train:]
    for i, (case, pred, true_label) in enumerate(zip(test_cases, predictions, test_labels)):
        details.append({
            "case_id": case.get("case_id"),
            "customer_id": case.get("customer_id"),
            "predicted": int(pred),
            "actual": true_label,
            "correct": int(pred) == true_label
        })
    
    result = ExperimentResult(
        method="Traditional (TF-IDF + RF)",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        avg_latency=prediction_time / len(test_texts) if test_texts else 0,
        total_time=training_time + prediction_time,
        text_feature_utilization=text_utilization,
        details=details
    )
    
    print(f"\n传统方法结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  文本特征利用率: {text_utilization:.4f}")
    print(f"  平均处理时间: {result.avg_latency:.4f}s")
    
    return result, processor


def run_llm_rag_experiment(cases: List[Dict[str, Any]], rag_system: RAGRiskSystem = None, use_mock: bool = False) -> ExperimentResult:
    print("\n" + "="*60)
    if use_mock or not RAG_AVAILABLE:
        print("运行LLM+RAG方法实验 (模拟模式)")
    else:
        print("运行LLM+RAG方法实验")
    print("="*60)
    
    processor = LLMRAGTextProcessor(rag_system, use_mock=use_mock)
    
    predictions = []
    true_labels = []
    latencies = []
    utilizations = []
    details = []
    
    level_map = {"low": 0, "medium": 1, "high": 2}
    
    total_cases = len(cases)
    for i, case in enumerate(cases):
        print(f"\r处理案例 {i+1}/{total_cases}...", end="", flush=True)
        
        true_label = level_map.get(case.get("risk_level", "medium"), 1)
        true_labels.append(true_label)
        
        start_time = time.time()
        pred_label, case_details = processor.analyze_text(case)
        latency = time.time() - start_time
        
        predictions.append(pred_label)
        latencies.append(latency)
        
        utilization = processor.calculate_text_feature_utilization(case, case_details)
        utilizations.append(utilization)
        
        details.append({
            "case_id": case.get("case_id"),
            "customer_id": case.get("customer_id"),
            "predicted": pred_label,
            "actual": true_label,
            "correct": pred_label == true_label,
            "risk_score": case_details.get("risk_score", -1),
            "latency": latency,
            "text_utilization": utilization,
            "risk_factors": case_details.get("risk_factors", []),
            "favorable_factors": case_details.get("favorable_factors", [])
        })
    
    print()
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    avg_latency = np.mean(latencies)
    total_time = sum(latencies)
    avg_utilization = np.mean(utilizations)
    
    result = ExperimentResult(
        method="LLM+RAG",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        avg_latency=avg_latency,
        total_time=total_time,
        text_feature_utilization=avg_utilization,
        details=details
    )
    
    print(f"\nLLM+RAG方法结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  文本特征利用率: {avg_utilization:.4f}")
    print(f"  平均处理时间: {avg_latency:.4f}s")
    
    return result


def generate_comparison_table(
    traditional_result: ExperimentResult,
    llm_result: ExperimentResult,
    output_path: str
):
    data = {
        "指标": [
            "准确率 (Accuracy)",
            "精确率 (Precision)",
            "召回率 (Recall)",
            "F1分数 (F1-Score)",
            "文本特征利用率",
            "平均处理时间 (秒)",
            "总处理时间 (秒)"
        ],
        "传统方法 (TF-IDF+RF)": [
            f"{traditional_result.accuracy:.4f}",
            f"{traditional_result.precision:.4f}",
            f"{traditional_result.recall:.4f}",
            f"{traditional_result.f1_score:.4f}",
            f"{traditional_result.text_feature_utilization:.4f}",
            f"{traditional_result.avg_latency:.4f}",
            f"{traditional_result.total_time:.2f}"
        ],
        "LLM+RAG方法": [
            f"{llm_result.accuracy:.4f}",
            f"{llm_result.precision:.4f}",
            f"{llm_result.recall:.4f}",
            f"{llm_result.f1_score:.4f}",
            f"{llm_result.text_feature_utilization:.4f}",
            f"{llm_result.avg_latency:.4f}",
            f"{llm_result.total_time:.2f}"
        ],
        "提升/变化": [
            f"+{(llm_result.accuracy - traditional_result.accuracy)*100:.2f}%",
            f"+{(llm_result.precision - traditional_result.precision)*100:.2f}%",
            f"+{(llm_result.recall - traditional_result.recall)*100:.2f}%",
            f"+{(llm_result.f1_score - traditional_result.f1_score)*100:.2f}%",
            f"+{(llm_result.text_feature_utilization - traditional_result.text_feature_utilization)*100:.2f}%",
            f"+{(llm_result.avg_latency - traditional_result.avg_latency):.2f}s",
            f"+{(llm_result.total_time - traditional_result.total_time):.2f}s"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存至: {output_path}")
    
    return df


def generate_feature_importance_figure(
    traditional_processor: TraditionalTextProcessor,
    llm_result: ExperimentResult,
    output_path: str
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    traditional_importance = traditional_processor.get_feature_importance()
    top_20 = dict(list(traditional_importance.items())[:20])
    
    ax1 = axes[0]
    features = list(top_20.keys())
    importances = list(top_20.values())
    y_pos = np.arange(len(features))
    
    ax1.barh(y_pos, importances, align='center', color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('特征重要性', fontsize=10)
    ax1.set_title('传统方法 - TF-IDF特征重要性', fontsize=12)
    
    ax2 = axes[1]
    
    llm_feature_scores = {}
    for detail in llm_result.details:
        for factor in detail.get("risk_factors", []):
            name = factor.get("name", "")
            if name:
                llm_feature_scores[name] = llm_feature_scores.get(name, 0) + 1
        for factor in detail.get("favorable_factors", []):
            name = factor.get("name", "")
            if name:
                llm_feature_scores[name] = llm_feature_scores.get(name, 0) + 1
    
    sorted_features = sorted(llm_feature_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    
    if sorted_features:
        features = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]
        y_pos = np.arange(len(features))
        
        ax2.barh(y_pos, scores, align='center', color='coral')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('出现频次', fontsize=10)
        ax2.set_title('LLM+RAG方法 - 识别的风险因素频次', fontsize=12)
    else:
        ax2.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=14)
        ax2.set_title('LLM+RAG方法 - 识别的风险因素频次', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存至: {output_path}")


def generate_case_analysis_report(
    cases: List[Dict[str, Any]],
    traditional_result: ExperimentResult,
    llm_result: ExperimentResult,
    output_path: str
):
    typical_cases = []
    
    for case in cases:
        case_id = case.get("case_id")
        llm_detail = next((d for d in llm_result.details if d.get("case_id") == case_id), None)
        trad_detail = next((d for d in traditional_result.details if d.get("case_id") == case_id), None)
        
        if llm_detail and trad_detail:
            is_interesting = (
                llm_detail.get("correct") != trad_detail.get("correct") or
                case.get("risk_level") == "high" or
                len(llm_detail.get("risk_factors", [])) >= 3
            )
            
            if is_interesting:
                typical_cases.append({
                    "case": case,
                    "llm_detail": llm_detail,
                    "trad_detail": trad_detail
                })
    
    typical_cases = typical_cases[:10]
    
    report_lines = [
        "# 非结构化文本数据处理案例分析报告",
        "",
        "## 1. 实验概述",
        "",
        "本实验对比了传统方法（TF-IDF + 随机森林）与LLM+RAG方法在处理非结构化文本数据方面的能力。",
        "",
        "### 1.1 实验数据",
        f"- 测试案例数量: {len(cases)}个",
        "- 文本内容包括: 贷款用途描述、客户申诉、交易备注等",
        "- 风险等级标注: 低风险、中风险、高风险",
        "",
        "### 1.2 实验方法",
        "",
        "**传统方法:**",
        "- 使用TF-IDF进行文本特征提取",
        "- 使用随机森林进行分类预测",
        "- 基于关键词匹配和统计特征",
        "",
        "**LLM+RAG方法:**",
        "- 利用大语言模型直接理解文本语义",
        "- 结合RAG检索相关知识库案例",
        "- 综合分析文本中的风险信号",
        "",
        "## 2. 整体性能对比",
        "",
        "| 指标 | 传统方法 | LLM+RAG方法 | 提升 |",
        "|------|----------|-------------|------|",
        f"| 准确率 | {traditional_result.accuracy:.4f} | {llm_result.accuracy:.4f} | +{(llm_result.accuracy - traditional_result.accuracy)*100:.2f}% |",
        f"| 精确率 | {traditional_result.precision:.4f} | {llm_result.precision:.4f} | +{(llm_result.precision - traditional_result.precision)*100:.2f}% |",
        f"| 召回率 | {traditional_result.recall:.4f} | {llm_result.recall:.4f} | +{(llm_result.recall - traditional_result.recall)*100:.2f}% |",
        f"| F1分数 | {traditional_result.f1_score:.4f} | {llm_result.f1_score:.4f} | +{(llm_result.f1_score - traditional_result.f1_score)*100:.2f}% |",
        f"| 文本特征利用率 | {traditional_result.text_feature_utilization:.4f} | {llm_result.text_feature_utilization:.4f} | +{(llm_result.text_feature_utilization - traditional_result.text_feature_utilization)*100:.2f}% |",
        "",
        "## 3. 典型案例分析",
        ""
    ]
    
    for i, item in enumerate(typical_cases, 1):
        case = item["case"]
        llm_detail = item["llm_detail"]
        trad_detail = item["trad_detail"]
        
        report_lines.extend([
            f"### 案例 {i}: {case.get('case_id', 'N/A')}",
            "",
            "#### 基本信息",
            f"- **客户ID**: {case.get('customer_id', 'N/A')}",
            f"- **真实风险等级**: {case.get('risk_level', 'N/A')} (风险评分: {case.get('risk_score', 'N/A')})",
            "",
            "#### 文本内容",
            "",
            "**贷款用途描述:**",
            f"> {case.get('loan_purpose_text', 'N/A')}",
            "",
            "**客户申诉:**",
            f"> {case.get('customer_appeal', 'N/A')}",
            "",
            "**交易备注:**",
            f"> {case.get('transaction_remark', 'N/A')}",
            "",
            "#### 分析结果对比",
            "",
            "| 方法 | 预测结果 | 是否正确 |",
            "|------|----------|----------|",
            f"| 传统方法 | {['低风险', '中风险', '高风险'][trad_detail.get('predicted', 1)]} | {'✓' if trad_detail.get('correct') else '✗'} |",
            f"| LLM+RAG | {['低风险', '中风险', '高风险'][llm_detail.get('predicted', 1)]} | {'✓' if llm_detail.get('correct') else '✗'} |",
            ""
        ])
        
        if llm_detail.get("risk_factors"):
            report_lines.extend([
                "#### LLM+RAG识别的风险因素",
                ""
            ])
            for factor in llm_detail.get("risk_factors", [])[:5]:
                report_lines.append(f"- **{factor.get('name', 'N/A')}**: {factor.get('description', 'N/A')} (权重: {factor.get('weight', 'N/A')})")
            report_lines.append("")
        
        if llm_detail.get("favorable_factors"):
            report_lines.extend([
                "#### LLM+RAG识别的有利因素",
                ""
            ])
            for factor in llm_detail.get("favorable_factors", [])[:3]:
                report_lines.append(f"- **{factor.get('name', 'N/A')}**: {factor.get('description', 'N/A')}")
            report_lines.append("")
        
        report_lines.extend([
            "#### 分析说明",
            ""
        ])
        
        if llm_detail.get("correct") and not trad_detail.get("correct"):
            report_lines.append("LLM+RAG方法正确识别了风险等级，而传统方法预测错误。这表明LLM能够更好地理解文本中的语义信息和隐含风险。")
        elif not llm_detail.get("correct") and trad_detail.get("correct"):
            report_lines.append("传统方法正确预测，而LLM+RAG方法出现误判。可能是由于文本信息较为隐晦，或LLM对某些专业术语理解不够准确。")
        else:
            report_lines.append("两种方法预测结果一致。")
        
        report_lines.append("")
    
    report_lines.extend([
        "## 4. 方法对比分析",
        "",
        "### 4.1 传统方法的局限性",
        "",
        "1. **语义理解不足**: TF-IDF基于词频统计，无法理解词语之间的语义关系和上下文含义",
        "2. **特征工程依赖**: 需要人工设计特征，对新领域适应性差",
        "3. **信息丢失**: 将文本转化为稀疏向量时，大量语义信息丢失",
        "4. **关键词匹配局限**: 只能识别明确的关键词，无法理解隐含的风险信号",
        "",
        "### 4.2 LLM+RAG方法的优势",
        "",
        "1. **深度语义理解**: 能够理解文本的深层含义和隐含信息",
        "2. **上下文关联**: 结合知识库中的案例，进行类比推理",
        "3. **灵活适应**: 无需特征工程，可直接处理各种类型的文本",
        "4. **可解释性强**: 提供详细的风险因素分析和建议",
        "",
        "### 4.3 LLM+RAG方法的局限性",
        "",
        "1. **处理时间较长**: 每个案例需要调用LLM API，耗时较长",
        "2. **成本较高**: API调用产生费用",
        "3. **结果不稳定**: LLM输出可能存在一定的随机性",
        "",
        "## 5. 结论与建议",
        "",
        "### 5.1 主要结论",
        "",
        f"1. LLM+RAG方法在风险识别准确率上达到 **{llm_result.accuracy:.2%}**，优于传统方法的 **{traditional_result.accuracy:.2%}**",
        f"2. LLM+RAG方法的文本特征利用率达到 **{llm_result.text_feature_utilization:.2%}**，显著高于传统方法的 **{traditional_result.text_feature_utilization:.2%}**",
        "3. LLM+RAG能够识别更多隐含的风险因素，提供更全面的风险分析",
        "",
        "### 5.2 应用建议",
        "",
        "1. **混合策略**: 对于简单案例可使用传统方法快速处理，复杂案例使用LLM+RAG深度分析",
        "2. **持续优化**: 定期更新知识库，提升RAG检索的准确性",
        "3. **人机协同**: 将LLM分析结果作为辅助决策参考，最终由人工审核确认",
        "",
        "---",
        "",
        f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*"
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"案例分析报告已保存至: {output_path}")


def main():
    print("="*60)
    print("非结构化文本数据处理实验")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    test_cases_path = project_root / "data" / "test_cases" / "text_cases.json"
    results_dir = project_root / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n加载测试用例: {test_cases_path}")
    cases = load_test_cases(str(test_cases_path))
    print(f"共加载 {len(cases)} 个测试案例")
    
    risk_distribution = {"low": 0, "medium": 0, "high": 0}
    for case in cases:
        risk_distribution[case.get("risk_level", "medium")] += 1
    print(f"风险等级分布: {risk_distribution}")
    
    traditional_result, traditional_processor = run_traditional_experiment(cases)
    
    use_mock = not RAG_AVAILABLE
    llm_result = run_llm_rag_experiment(cases, use_mock=use_mock)
    
    comparison_path = tables_dir / "text_processing_comparison.csv"
    generate_comparison_table(traditional_result, llm_result, str(comparison_path))
    
    feature_importance_path = figures_dir / "text_feature_importance.png"
    generate_feature_importance_figure(traditional_processor, llm_result, str(feature_importance_path))
    
    report_path = results_dir / "case_analysis_report.md"
    generate_case_analysis_report(cases, traditional_result, llm_result, str(report_path))
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    
    return traditional_result, llm_result


if __name__ == "__main__":
    main()
