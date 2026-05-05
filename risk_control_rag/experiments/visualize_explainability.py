"""
可解释性雷达图生成模块

生成不同模型的可解释性评分雷达图对比。
"""

import sys
import os
import json
import math
from typing import List, Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config.evaluation_config import ExplainabilityDimension


def create_radar_chart(
    scores_by_model: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "可解释性评分雷达图对比",
    figsize: tuple = (10, 8)
):
    """
    创建雷达图对比不同模型的可解释性评分
    
    Args:
        scores_by_model: 各模型的评分字典，格式为 {模型名: {维度名: 分数}}
        output_path: 输出图片路径
        title: 图表标题
        figsize: 图表大小
    """
    dimensions = ["逻辑清晰度", "证据充分性", "可理解性", "完整性", "专业性"]
    num_dims = len(dimensions)
    
    angles = [n / float(num_dims) * 2 * math.pi for n in range(num_dims)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    for idx, (model_name, scores) in enumerate(scores_by_model.items()):
        values = [scores.get(dim, 0) for dim in dimensions]
        values += values[:1]
        
        color = colors[idx % len(colors)]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=12)
    
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"雷达图已保存至: {output_path}")


def create_bar_chart(
    scores_by_model: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "可解释性评分柱状图对比",
    figsize: tuple = (12, 6)
):
    """
    创建柱状图对比不同模型的可解释性评分
    """
    dimensions = ["逻辑清晰度", "证据充分性", "可理解性", "完整性", "专业性"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(dimensions))
    width = 0.8 / len(scores_by_model)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    for idx, (model_name, scores) in enumerate(scores_by_model.items()):
        values = [scores.get(dim, 0) for dim in dimensions]
        offset = (idx - len(scores_by_model) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, 
                     color=colors[idx % len(colors)], alpha=0.8)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('评估维度', fontsize=12)
    ax.set_ylabel('评分 (1-5)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"柱状图已保存至: {output_path}")


def create_heatmap(
    scores_by_model: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "可解释性评分热力图",
    figsize: tuple = (10, 6)
):
    """
    创建热力图展示评分分布
    """
    dimensions = ["逻辑清晰度", "证据充分性", "可理解性", "完整性", "专业性"]
    
    model_names = list(scores_by_model.keys())
    data = np.array([[scores_by_model[m].get(d, 0) for d in dimensions] 
                     for m in model_names])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
    
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_yticklabels(model_names, fontsize=11)
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    for i in range(len(model_names)):
        for j in range(len(dimensions)):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha='center', va='center', fontsize=11,
                          color='white' if data[i, j] < 2.5 or data[i, j] > 4 else 'black')
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('评分', fontsize=11)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"热力图已保存至: {output_path}")


def create_risk_level_comparison(
    results_by_risk_level: Dict[str, Dict[str, float]],
    output_path: str,
    figsize: tuple = (10, 6)
):
    """
    创建不同风险等级的评分对比图
    """
    dimensions = ["逻辑清晰度", "证据充分性", "可理解性", "完整性", "专业性"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(dimensions))
    width = 0.25
    
    colors = {'低风险': '#28a745', '中风险': '#ffc107', '高风险': '#dc3545'}
    
    for idx, (risk_level, scores) in enumerate(results_by_risk_level.items()):
        values = [scores.get(dim, 0) for dim in dimensions]
        bars = ax.bar(x + idx * width, values, width, label=risk_level,
                     color=colors.get(risk_level, '#666666'), alpha=0.8)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('评估维度', fontsize=12)
    ax.set_ylabel('评分 (1-5)', fontsize=12)
    ax.set_title('不同风险等级的可解释性评分对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"风险等级对比图已保存至: {output_path}")


def load_scores_from_csv(csv_path: str) -> Dict[str, Dict[str, float]]:
    """从CSV加载评分数据"""
    import csv
    
    scores_by_model = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['模型类型']
            if model not in scores_by_model:
                scores_by_model[model] = {
                    "逻辑清晰度": [],
                    "证据充分性": [],
                    "可理解性": [],
                    "完整性": [],
                    "专业性": [],
                }
            
            scores_by_model[model]["逻辑清晰度"].append(float(row['逻辑清晰度']))
            scores_by_model[model]["证据充分性"].append(float(row['证据充分性']))
            scores_by_model[model]["可理解性"].append(float(row['可理解性']))
            scores_by_model[model]["完整性"].append(float(row['完整性']))
            scores_by_model[model]["专业性"].append(float(row['专业性']))
    
    avg_scores = {}
    for model, scores in scores_by_model.items():
        avg_scores[model] = {
            dim: sum(vals) / len(vals) for dim, vals in scores.items()
        }
    
    return avg_scores


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    
    demo_scores = {
        "RAG": {
            "逻辑清晰度": 4.2,
            "证据充分性": 4.5,
            "可理解性": 4.0,
            "完整性": 4.3,
            "专业性": 4.4,
        },
        "LLM-Only": {
            "逻辑清晰度": 3.8,
            "证据充分性": 3.2,
            "可理解性": 4.1,
            "完整性": 3.5,
            "专业性": 3.9,
        },
        "Traditional-ML": {
            "逻辑清晰度": 2.5,
            "证据充分性": 2.0,
            "可理解性": 2.8,
            "完整性": 2.2,
            "专业性": 2.5,
        },
    }
    
    radar_path = project_root / "results" / "figures" / "explainability_radar.png"
    create_radar_chart(demo_scores, str(radar_path))
    
    bar_path = project_root / "results" / "figures" / "explainability_bar.png"
    create_bar_chart(demo_scores, str(bar_path))
    
    heatmap_path = project_root / "results" / "figures" / "explainability_heatmap.png"
    create_heatmap(demo_scores, str(heatmap_path))
    
    risk_level_scores = {
        "低风险": {
            "逻辑清晰度": 4.5,
            "证据充分性": 4.6,
            "可理解性": 4.3,
            "完整性": 4.4,
            "专业性": 4.5,
        },
        "中风险": {
            "逻辑清晰度": 4.1,
            "证据充分性": 4.3,
            "可理解性": 4.0,
            "完整性": 4.2,
            "专业性": 4.3,
        },
        "高风险": {
            "逻辑清晰度": 3.8,
            "证据充分性": 4.0,
            "可理解性": 3.7,
            "完整性": 3.9,
            "专业性": 4.1,
        },
    }
    
    risk_comparison_path = project_root / "results" / "figures" / "risk_level_comparison.png"
    create_risk_level_comparison(risk_level_scores, str(risk_comparison_path))
    
    print("\n所有图表生成完成!")


if __name__ == "__main__":
    main()
