# Risk Control RAG

基于检索增强生成(RAG)的风控系统实验项目。

## 项目结构

```
risk_control_rag/
├── config/                 # 配置文件
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── knowledge_base/    # 知识库数据
├── models/                 # 模型目录
│   ├── baseline/          # 基线模型
│   └── rag/               # RAG模型
├── experiments/            # 实验脚本
├── utils/                  # 工具函数
├── results/                # 实验结果
│   ├── tables/            # 结果表格
│   └── figures/           # 结果图表
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量，创建 `.env` 文件：
```
VOLCENGINE_ACCESS_KEY=your_access_key
VOLCENGINE_SECRET_KEY=your_secret_key
VOLCENGINE_ENDPOINT_ID=your_endpoint_id
```

## 使用方法

```bash
# 运行主实验
python experiments/main_experiment.py

# 运行消融实验
python experiments/ablation.py

# 运行评估
python experiments/evaluation.py
```

## 模型说明

- **基线模型**: Random Forest, Gradient Boosting, XGBoost
- **RAG模型**: 基于火山引擎LLM和Embedding的检索增强生成模型
