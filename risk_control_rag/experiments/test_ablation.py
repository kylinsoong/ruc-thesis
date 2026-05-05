"""
模拟消融实验测试脚本

使用模拟数据测试消融实验流程，不实际调用API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.ablation import (
    AblationExperiment, 
    ABLATION_CONFIGS,
    AblationResult,
)


def run_mock_ablation():
    """运行模拟消融实验"""
    
    mock_results = {
        'full': AblationResult(
            config_name='full',
            total_samples=20,
            correct_predictions=16,
            accuracy=0.80,
            avg_latency=3.5,
            total_tokens=45000,
            avg_tokens=2250,
            total_cost=0.45,
            avg_cost=0.0225,
            explainability_score=75.5,
            risk_factors_avg=3.2,
            knowledge_sources_avg=4.5,
            parse_success_rate=0.95,
            low_risk_count=6,
            medium_risk_count=8,
            high_risk_count=5,
            unknown_risk_count=1,
        ),
        'llm_only': AblationResult(
            config_name='llm_only',
            total_samples=20,
            correct_predictions=12,
            accuracy=0.60,
            avg_latency=2.1,
            total_tokens=38000,
            avg_tokens=1900,
            total_cost=0.38,
            avg_cost=0.019,
            explainability_score=35.0,
            risk_factors_avg=1.5,
            knowledge_sources_avg=0,
            parse_success_rate=0.85,
            low_risk_count=7,
            medium_risk_count=6,
            high_risk_count=4,
            unknown_risk_count=3,
        ),
        'bm25_only': AblationResult(
            config_name='bm25_only',
            total_samples=20,
            correct_predictions=14,
            accuracy=0.70,
            avg_latency=3.2,
            total_tokens=42000,
            avg_tokens=2100,
            total_cost=0.42,
            avg_cost=0.021,
            explainability_score=60.0,
            risk_factors_avg=2.8,
            knowledge_sources_avg=3.5,
            parse_success_rate=0.90,
            low_risk_count=5,
            medium_risk_count=9,
            high_risk_count=5,
            unknown_risk_count=1,
        ),
        'vector_only': AblationResult(
            config_name='vector_only',
            total_samples=20,
            correct_predictions=15,
            accuracy=0.75,
            avg_latency=3.0,
            total_tokens=43000,
            avg_tokens=2150,
            total_cost=0.43,
            avg_cost=0.0215,
            explainability_score=68.5,
            risk_factors_avg=3.0,
            knowledge_sources_avg=4.0,
            parse_success_rate=0.92,
            low_risk_count=6,
            medium_risk_count=8,
            high_risk_count=5,
            unknown_risk_count=1,
        ),
    }
    
    experiment = AblationExperiment(output_dir='results')
    experiment.results = mock_results
    
    print('保存CSV结果...')
    experiment.save_results_csv()
    
    print('保存详细结果...')
    experiment.save_detailed_results()
    
    print('生成对比图...')
    experiment.plot_comparison()
    
    print('生成报告...')
    report = experiment.generate_report()
    
    print('\n' + '='*60)
    print('模拟实验完成！')
    print('='*60)
    print('\n生成的文件:')
    print('  - results/tables/ablation_results.csv')
    print('  - results/tables/ablation_detailed_results.json')
    print('  - results/figures/ablation_comparison.png')
    print('  - results/ablation_report.md')
    
    return experiment.results


if __name__ == '__main__':
    run_mock_ablation()
