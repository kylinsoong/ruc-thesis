# 金融风控系统实验手册

本文档为基于大语言模型（Large Language Model，LLM）与检索增强生成（Retrieval-Augmented Generation，RAG）技术的金融风控系统研究论文提供详细的实验操作指南。通过本手册，读者可按照步骤逐步复现论文中的实验结果，深入理解LLM与RAG技术在金融风控领域的应用方法。

---

## 第1章 实验准备

本章详细介绍进行金融风控实验所需的硬件环境、软件依赖、API配置以及项目代码结构，确保实验环境的一致性和可复现性。

### 1.1 环境要求

#### 1.1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | Intel Core i5 第8代 | Intel Core i7-12700H 或以上 |
| 内存 | 16 GB | 32 GB DDR4 |
| 硬盘 | 50 GB 可用空间 | 1 TB NVMe SSD |
| 网络 | 稳定互联网连接 | 带宽 10Mbps 以上 |

**说明**：由于本研究采用云API方式调用大语言模型，因此无需本地GPU资源。实测网络延迟约为20-50ms，满足实验需求。

#### 1.1.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| 操作系统 | macOS 12.0+ / Ubuntu 20.04+ / Windows 10+ | 推荐使用macOS或Linux |
| Python | 3.9.x | 建议使用Python 3.9.18 |
| Git | 2.30+ | 版本控制 |

### 1.2 软件依赖

#### 1.2.1 核心依赖安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装核心依赖
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install xgboost==2.0.0
pip install torch==2.0.1
pip install chromadb==0.4.15
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
```

#### 1.2.2 辅助依赖安装

```bash
pip install requests==2.31.0
pip install python-dotenv==1.0.0
pip install scipy==1.11.0
```

### 1.3 API配置

#### 1.3.1 豆包API申请

1. 访问[火山引擎官网](https://www.volcengine.com/)注册账号
2. 申请访问密钥（Access Key 和 Secret Key）
3. 在控制台创建应用，获取API调用权限

#### 1.3.2 环境变量配置

```bash
# 创建 .env 文件
cat > .env << EOF
ARK_API_KEY=your_access_key_here
ARK_SECRET_KEY=your_secret_key_here
ARK_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3
EOF
```

#### 1.3.3 API客户端封装

```python
# api_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class DoubaoAPIClient:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        self.secret_key = os.getenv("ARK_SECRET_KEY")
        self.endpoint = os.getenv("ARK_ENDPOINT")
    
    def call_llm(self, prompt, model="doubao-seed-2-0-pro-260215", 
                  temperature=0.3, max_tokens=4096):
        """调用豆包大语言模型"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            json=payload
        )
        return response.json()
    
    def get_embedding(self, text, model="doubao-embedding-vision-251215"):
        """获取文本Embedding向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": model, "input": text}
        response = requests.post(
            f"{self.endpoint}/embeddings",
            headers=headers,
            json=payload
        )
        return response.json()
```

#### 1.3.4 API调用测试

```python
# test_api.py
from api_client import DoubaoAPIClient

client = DoubaoAPIClient()

# 测试LLM调用
result = client.call_llm("你好，请做自我介绍")
print(result)

# 测试Embedding调用
embedding = client.get_embedding("这是一条测试文本")
print(embedding)
```

**预期结果**：
- LLM调用返回包含 `choices[0].message.content` 的JSON响应
- Embedding调用返回 2048 维向量

### 1.4 代码结构

```
project/
├── data/
│   ├── raw/                 # 原始数据集
│   │   ├── credit_card.csv
│   │   └── german_credit.csv
│   └── processed/           # 处理后数据
├── knowledge_base/          # 知识库文档
│   ├── risk_factors.txt
│   ├── credit_rules.txt
│   └── ...
├── models/                  # 模型实现
│   ├── lr_model.py
│   ├── rf_model.py
│   ├── xgboost_model.py
│   ├── deepfm_model.py
│   └── tabnet_model.py
├── rag/                     # RAG系统
│   ├── retriever.py
│   ├── embedder.py
│   ├── bm25.py
│   └── hybrid_search.py
├── evaluation/              # 评估模块
│   ├── metrics.py
│   └── explainability.py
├── api_client.py           # API客户端
├── config.py               # 配置文件
├── requirements.txt
└── EXPERIMENT_HANDBOOK.md
```

---

## 第2章 数据集准备

本章详细介绍实验所用数据集的获取方式、预处理步骤以及知识库的构建方法。

### 2.1 Credit Card Fraud数据集

#### 2.1.1 数据集获取

**来源**：Kaggle平台 - Credit Card Fraud Detection数据集

**下载地址**：
```bash
# 下载数据集（需要Kaggle账号）
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

**数据统计**：
| 统计项 | 数值 |
|--------|------|
| 总样本数 | 284,807（原始）/ 10,000（实验子集） |
| 特征数量 | 30（Time, Amount, V1-V28） |
| 正常交易 | 99.83% |
| 欺诈交易 | 0.17% |
| 不平衡比 | 约587:1 |

#### 2.1.2 数据预处理

```python
# preprocess_credit.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 加载数据
df = pd.read_csv("data/raw/creditcard.csv")

# 采样获得实验子集（保持类别比例）
df_sample = df.sample(n=10000, random_state=42)

# 特征与标签分离
X = df_sample.drop(["Class", "Time"], axis=1)
y = df_sample["Class"]

# 划分训练集和测试集（分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 标准化Amount特征
scaler = StandardScaler()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"] = scaler.transform(X_test[["Amount"]])

# SMOTE过采样（仅对训练集）
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 保存处理后数据
X_train_resampled.to_csv("data/processed/credit_train.csv", index=False)
X_test.to_csv("data/processed/credit_test.csv", index=False)
```

**前置条件**：
- 已下载Credit Card Fraud数据集
- 已安装 imblearn：`pip install imbalanced-learn`

**预期结果**：
- 训练集约16,000条样本（含SMOTE增强）
- 测试集2,000条样本
- 类别不平衡得到缓解

**验证方法**：
```python
print(f"训练集样本数: {len(X_train_resampled)}")
print(f"正类比例: {y_train_resampled.mean():.2%}")
print(f"测试集样本数: {len(X_test)}")
```

### 2.2 German Credit数据集

#### 2.2.1 数据集获取

**来源**：UCI机器学习仓库 - German Credit Data

**下载地址**：
```bash
# 方法1：直接下载
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

# 方法2：使用pandas
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = [f"attr_{i}" for i in range(20)] + ["credit_status"]
df = pd.read_csv(url, sep=" ", header=None, names=columns)
```

**数据统计**：
| 统计项 | 数值 |
|--------|------|
| 总样本数 | 1,000 |
| 特征数量 | 20（7个数值特征，13个类别特征） |
| 好客户 | 700（70%） |
| 坏客户 | 300（30%） |
| 不平衡比 | 约2.3:1 |

#### 2.2.2 数据预处理

```python
# preprocess_german.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 加载数据
df = pd.read_csv("data/raw/german_credit.csv")

# 分离特征和标签
X = df.drop("credit_status", axis=1)
y = df["credit_status"]

# 对类别特征进行Label Encoding
categorical_cols = X.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# 对数值特征进行标准化
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# 划分训练集和测试集（分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 保存处理后数据
X_train.to_csv("data/processed/german_train.csv", index=False)
X_test.to_csv("data/processed/german_test.csv", index=False)
```

**前置条件**：
- 已获取German Credit数据集
- 已安装 scikit-learn

**预期结果**：
- 训练集800条，测试集200条
- 13个类别特征完成编码
- 7个数值特征完成标准化

**验证方法**：
```python
print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"特征维度: {X_train.shape[1]}")
```

### 2.3 知识库构建

#### 2.3.1 知识文档准备

创建金融风控领域知识文档：

```bash
mkdir -p knowledge_base
```

**knowledge_base/risk_factors.txt**：
```
金融风控关键风险因素

1. 信用历史
   - 还款记录良好的申请人违约概率较低
   - 逾期次数越多，违约风险越高
   - 历史违约记录是最强的预测因子之一

2. 负债情况
   - 负债收入比（Debt-to-Income Ratio）是核心指标
   - 建议阈值：负债收入比不应超过36%
   - 信用卡透支额度使用率过高表示资金紧张

3. 信用查询记录
   - 近期频繁申请贷款表示资金需求旺盛
   - 硬查询过多负面影响信用评分
   - 建议审查时间窗口：6-12个月
```

**knowledge_base/credit_rules.txt**：
```
信贷审批规则

1. 审批标准
   - 信用评分达到650分以上可考虑批准
   - 收入稳定性需满足6个月以上
   - 年龄要求：18岁以上，65岁以下

2. 风险等级划分
   - 低风险：多项指标优良，无负面信号
   - 中风险：部分指标存在瑕疵，需人工审核
   - 高风险：多项指标异常，建议拒绝

3. 决定因素
   - 主要考虑：还款能力、信用历史、负债水平
   - 次要考虑：职业稳定性、居住稳定性
```

#### 2.3.2 构建向量知识库

```python
# build_knowledge_base.py
import chromadb
from chromadb.config import Settings
import os

class KnowledgeBaseBuilder:
    def __init__(self, persist_dir="knowledge_base/chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
    
    def load_documents(self, doc_dir="knowledge_base"):
        """加载知识文档"""
        documents = []
        metadata = []
        ids = []
        
        for filename in os.listdir(doc_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(doc_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    metadata.append({"source": filename})
                    ids.append(f"doc_{len(ids)}")
        
        return documents, metadata, ids
    
    def build_vector_store(self, embeddings):
        """构建向量数据库"""
        collection = self.client.get_or_create_collection(
            name="financial_knowledge",
            metadata={"description": "金融风控领域知识库"}
        )
        
        documents, metadata, ids = self.load_documents()
        
        collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids,
            embeddings=embeddings
        )
        
        return collection
    
    def query(self, query_text, n_results=5):
        """查询相关知识"""
        collection = self.client.get_collection("financial_knowledge")
        return collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

# 使用示例
builder = KnowledgeBaseBuilder()
collection = builder.build_vector_store(embeddings=your_embeddings)
results = builder.query("信用评分低如何评估风险")
```

**前置条件**：
- 已创建知识文档
- 已配置豆包API（用于生成Embeddings）

**预期结果**：
- ChromaDB持久化存储
- 25个知识文档被索引
- 可进行相似性检索

**验证方法**：
```python
collection = client.get_collection("financial_knowledge")
print(f"知识库文档数量: {collection.count()}")
```

---

## 第3章 基线模型实验

本章详细介绍六个基线模型的实验步骤，包括Logistic Regression、Random Forest、XGBoost、DeepFM、TabNet以及LLM基线。

### 3.1 Logistic Regression

#### 3.1.1 模型实现

```python
# lr_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(
            penalty="l2",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    
    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """预测"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """预测概率"""
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_proba)
        }

# 训练脚本
if __name__ == "__main__":
    # 加载数据
    X_train = pd.read_csv("data/processed/german_train.csv")
    X_test = pd.read_csv("data/processed/german_test.csv")
    
    # 加载标签
    train_df = pd.read_csv("data/raw/german_credit.csv")
    y = train_df["credit_status"]
    _, y_test = train_test_split(y, test_size=0.2, stratify=y, random_state=42)
    y_train = y.iloc[:len(X_train)]
    
    # 训练模型
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    
    # 评估模型
    results = model.evaluate(X_test, y_test)
    print("Logistic Regression Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
```

**前置条件**：
- 已完成数据集预处理
- 已安装 scikit-learn

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.715 | 0.854 |
| F1 | 0.599 | 0.877 |
| AUC-ROC | 0.790 | 0.903 |

**验证方法**：
```bash
python lr_model.py
```

### 3.2 Random Forest

#### 3.2.1 模型实现

```python
# rf_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_proba)
        }
    
    def feature_importance(self, feature_names):
        """获取特征重要性"""
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))

# 训练脚本
if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/german_train.csv")
    X_test = pd.read_csv("data/processed/german_test.csv")
    
    model = RandomForestModel(n_estimators=100)
    model.train(X_train, y_train)
    
    results = model.evaluate(X_test, y_test)
    print("Random Forest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
```

**前置条件**：同3.1.1

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.770 | 0.994 |
| F1 | 0.547 | 0.851 |
| AUC-ROC | 0.808 | 0.998 |

### 3.3 XGBoost

#### 3.3.1 模型实现

```python
# xgboost_model.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_proba)
        }
```

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.775 | 0.998 |
| F1 | 0.533 | 0.883 |
| AUC-ROC | 0.796 | 0.999 |

### 3.4 DeepFM

#### 3.4.1 模型实现

```python
# deepfm_model.py
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class DeepFM(nn.Module):
    def __init__(self, feature_dim, embed_dim=16):
        super(DeepFM, self).__init__()
        
        # FM组件
        self.linear = nn.Linear(feature_dim, 1)
        self.embedding = nn.Embedding(feature_dim, embed_dim)
        
        # Deep组件
        self.deep = nn.Sequential(
            nn.Linear(feature_dim * embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.output = nn.Linear(64 + 1, 1)
    
    def forward(self, x):
        # Linear part
        linear_out = self.linear(x)
        
        # FM part
        embed = self.embedding(x)
        fm_out = 0.5 * (embed.sum(dim=1) ** 2 - embed.sum(dim=1) ** 2)
        
        # Deep part
        deep_out = self.deep(embed.view(x.size(0), -1))
        
        # Combine
        out = torch.cat([linear_out, fm_out, deep_out], dim=1)
        return torch.sigmoid(self.output(out))

# 训练脚本略
```

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.770 | 1.000 |
| F1 | 0.591 | 0.905 |
| AUC-ROC | 0.792 | 1.000 |

### 3.5 TabNet

#### 3.5.1 模型实现

```python
# tabnet_model.py
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class TabNetModel:
    def __init__(self, n_d=32, n_a=32, n_steps=5):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        # 实际实现建议使用pytorch-tabnet库
        # from pytorch_tabnet import TabNetClassifier
        # self.model = TabNetClassifier(
        #     n_d=n_d, n_a=n_a, n_steps=n_steps,
        #     gamma=1.5, optimizer_fn=torch.optim.Adam
        # )
    
    def train(self, X_train, y_train, max_epochs=100):
        # 训练代码略
        pass
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, self.predict_proba(X_test))
        }
```

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.770 | 0.999 |
| F1 | 0.592 | 0.868 |
| AUC-ROC | 0.792 | 0.999 |

---

## 第4章 LLM基线实验

本章介绍如何配置和使用大语言模型进行金融风控基线实验。

### 4.1 API调用配置

#### 4.1.1 完整API客户端

```python
# llm_client.py
import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class FinancialRiskLLM:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        self.secret_key = os.getenv("ARK_SECRET_KEY")
        self.endpoint = os.getenv("ARK_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3")
        self.model = "doubao-seed-2-0-pro-260215"
        self.temperature = 0.3
        self.max_tokens = 2048
    
    def build_prompt(self, feature_data, feature_names):
        """构建风险评估Prompt"""
        feature_str = "\n".join([
            f"- {name}: {value}" 
            for name, value in zip(feature_names, feature_data)
        ])
        
        prompt = f"""你是一位资深的金融风控专家。请根据以下客户特征数据进行风险评估。

## 客户特征
{feature_str}

## 评估要求
1. 分析各特征的风险信号
2. 给出风险等级（低/中/高）
3. 提供简要的风险理由
4. 给出审批建议（批准/拒绝/需要人工审核）

## 输出格式
请严格按照以下JSON格式输出：
{{
    "risk_level": "低/中/高",
    "risk_score": 0.0到1.0之间的风险评分,
    "risk_factors": ["风险因素1", "风险因素2"],
    "favorable_factors": ["有利因素1"],
    "recommendation": "批准/拒绝/需要人工审核"
}}
"""
        return prompt
    
    def predict(self, feature_data, feature_names):
        """单条预测"""
        prompt = self.build_prompt(feature_data, feature_names)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            json=payload
        )
        latency = time.time() - start_time
        
        result = response.json()
        return {
            "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "latency": latency,
            "tokens_used": result.get("usage", {}).get("total_tokens", 0)
        }
    
    def batch_predict(self, X_test, feature_names, sample_size=50):
        """批量预测"""
        results = []
        indices = X_test.sample(n=min(sample_size, len(X_test)), random_state=42).index
        
        for idx in indices:
            feature_data = X_test.loc[idx].values
            result = self.predict(feature_data, feature_names)
            results.append(result)
            time.sleep(0.5)  # 避免API限流
        
        return results
```

#### 4.1.2 API调用测试

```python
# test_llm.py
from llm_client import FinancialRiskLLM
import pandas as pd

client = FinancialRiskLLM()

# 加载测试数据
X_test = pd.read_csv("data/processed/german_test.csv")
feature_names = X_test.columns.tolist()

# 单条测试
sample = X_test.iloc[0].values
result = client.predict(sample, feature_names)
print(f"LLM响应: {result['response']}")
print(f"延迟: {result['latency']:.2f}秒")
```

**前置条件**：
- 已配置豆包API密钥
- 已完成数据预处理

**预期结果**：
- API调用成功返回JSON格式响应
- 平均响应延迟2.5-4.0秒
- 风险等级判定合理

### 4.2 Prompt设计

#### 4.2.1 默认Prompt模板

```python
DEFAULT_PROMPT_TEMPLATE = """你是一位资深的金融风控专家，拥有20年金融风控经验。请根据以下客户特征数据进行专业的风险评估。

## 客户特征
{feature_str}

## 背景知识
{context_str}

## 评估要求
1. 仔细分析每一个特征指标
2. 识别关键风险信号和有利信号
3. 结合领域知识进行综合判断
4. 给出明确的风险等级和审批建议

## 输出格式
请严格按照以下JSON格式输出，确保可以被我程序解析：
{{
    "risk_level": "低/中/高",
    "risk_score": 0.0到1.0之间的风险评分,
    "risk_factors": ["风险因素1", "风险因素2"],
    "favorable_factors": ["有利因素1", "有利因素2"],
    "recommendation": "批准/拒绝/需要人工审核",
    "confidence": 0.0到1.0之间的置信度
}}

请直接输出JSON，不要有其他内容。
"""
```

#### 4.2.2 精简Prompt模板

```python
CONCISE_PROMPT_TEMPLATE = """风控专家评估：{feature_str}
输出JSON：{{"risk_level": "低/中/高", "risk_score": 0-1, "recommendation": "批准/拒绝/人工"}}
"""
```

### 4.3 结果评估

#### 4.3.1 LLM结果解析

```python
# parse_llm_response.py
import json
import re

def parse_llm_response(response_text):
    """解析LLM响应文本"""
    try:
        # 尝试提取JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        
        # 备用解析方法
        return {
            "risk_level": "未知",
            "error": "无法解析响应"
        }
    except Exception as e:
        return {
            "risk_level": "未知",
            "error": str(e)
        }

def calculate_metrics(results, y_true):
    """计算评估指标"""
    y_pred = []
    for r, true_label in zip(results, y_true):
        parsed = parse_llm_response(r["response"])
        pred_label = 1 if parsed.get("risk_level") == "高" else 0
        y_pred.append(pred_label)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
```

#### 4.3.2 评估脚本

```python
# evaluate_llm.py
from llm_client import FinancialRiskLLM
from parse_llm_response import parse_llm_response, calculate_metrics
import pandas as pd

# 加载数据
X_test = pd.read_csv("data/processed/german_test.csv")
y_test = pd.read_csv("data/raw/german_credit.csv")["credit_status"].iloc[-len(X_test):]

client = FinancialRiskLLM()
feature_names = X_test.columns.tolist()

# 批量预测
results = client.batch_predict(X_test, feature_names, sample_size=50)

# 计算指标
metrics = calculate_metrics(results, y_test.values)
print("LLM基线结果（German Credit）：")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

**前置条件**：
- 已完成LLM API配置
- 已完成数据预处理

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.650 | 0.723 |
| F1 | 0.552 | 0.616 |
| AUC-ROC | 0.720 | 0.752 |

**验证方法**：
```bash
python evaluate_llm.py
```

---

## 第5章 LLM+RAG主实验

本章介绍完整的LLM+RAG金融风控系统的实现步骤。

### 5.1 检索系统配置

#### 5.1.1 向量数据库初始化

```python
# vector_store.py
import chromadb
from chromadb.config import Settings
import numpy as np

class VectorStoreManager:
    def __init__(self, persist_dir="knowledge_base/chroma"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        self.collection = None
    
    def create_collection(self, name="financial_risk", dim=2048):
        """创建向量集合"""
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"dimension": dim}
        )
        return self.collection
    
    def add_documents(self, documents, embeddings, metadatas=None, ids=None):
        """添加文档"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        self.client.persist()
    
    def query(self, query_embedding, n_results=5):
        """查询相似文档"""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    
    def get_collection_info(self):
        """获取集合信息"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "dimension": self.collection.metadata.get("dimension")
        }
```

#### 5.1.2 Embedding服务

```python
# embedder.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        self.endpoint = os.getenv("ARK_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3")
        self.model = "doubao-embedding-vision-251215"
        self.dimension = 2048
    
    def embed_text(self, text):
        """获取单条文本的Embedding"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": text
        }
        
        response = requests.post(
            f"{self.endpoint}/embeddings",
            headers=headers,
            json=payload
        )
        
        result = response.json()
        return result.get("data", [{}])[0].get("embedding", [])
    
    def embed_batch(self, texts, batch_size=32):
        """批量获取Embedding"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                emb = self.embed_text(text)
                embeddings.append(emb)
        return embeddings
```

### 5.2 混合检索实现

#### 5.2.1 BM25检索器

```python
# bm25.py
import math
from collections import Counter

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
    
    def fit(self, documents):
        """构建BM25索引"""
        self.documents = documents
        N = len(documents)
        
        # 计算词频
        self.doc_freqs = []
        for doc in documents:
            freq = Counter(doc.split())
            self.doc_freqs.append(freq)
        
        # 计算 IDF
        df = Counter()
        for doc in self.doc_freqs:
            for word in doc.keys():
                df[word] += 1
        
        for word, freq in df.items():
            self.idf[word] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        
        # 计算平均文档长度
        self.avgdl = sum(len(doc.split()) for doc in documents) / N
    
    def score(self, query, doc_idx):
        """计算单个文档的BM25分数"""
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = len(self.documents[doc_idx].split())
        
        score = 0
        for term in query.split():
            if term not in doc_freq:
                continue
            
            tf = doc_freq[term]
            idf = self.idf.get(term, 0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def retrieve(self, query, top_k=5):
        """检索最相关的文档"""
        scores = []
        for i in range(len(self.documents)):
            scores.append((i, self.score(query, i)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

#### 5.2.2 RRF融合算法

```python
# hybrid_search.py
def reciprocal_rank_fusion(results_list, k=60):
    """
    RRF融合算法
    
    参数:
        results_list: 多个检索结果列表，每个列表包含(doc_id, score)元组
        k: RRF参数，默认60
    
    返回:
        融合后的结果列表
    """
    scores = {}
    
    for results in results_list:
        for rank, (doc_id, score) in enumerate(results, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0
            # RRF分数 + 原始相似度加权
            scores[doc_id] += 1.0 / (k + rank) + score * 0.1
    
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, rrf_k=60):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
    
    def retrieve(self, query, top_k=5):
        """混合检索"""
        # 向量检索
        query_embedding = embedder.embed_text(query)
        vector_results = self.vector_store.query(query_embedding, top_k * 2)
        
        # BM25检索
        bm25_results = self.bm25.retrieve(query, top_k * 2)
        
        # RRF融合
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.rrf_k
        )
        
        return fused_results[:top_k]
```

### 5.3 RAG Pipeline运行

#### 5.3.1 完整RAG系统

```python
# rag_system.py
class FinancialRiskRAGSystem:
    def __init__(self, llm_client, retriever, embedder):
        self.llm = llm_client
        self.retriever = retriever
        self.embedder = embedder
    
    def build_prompt(self, feature_data, feature_names, retrieved_knowledge):
        """构建增强Prompt"""
        feature_str = "\n".join([
            f"- {name}: {value}" 
            for name, value in zip(feature_names, feature_data)
        ])
        
        knowledge_str = "\n".join([
            f"[{i+1}] {doc}" 
            for i, doc in enumerate(retrieved_knowledge)
        ])
        
        prompt = f"""你是一位资深的金融风控专家，拥有20年金融风控经验。请根据以下客户特征数据和专业知识进行风险评估。

## 客户特征
{feature_str}

## 相关专业知识
{knowledge_str}

## 评估要求
1. 结合专业知识分析各特征的风险信号
2. 参考相似案例进行判断
3. 给出明确的风险等级和审批建议

## 输出格式
请严格按照以下JSON格式输出：
{{
    "risk_level": "低/中/高",
    "risk_score": 0.0到1.0之间的风险评分,
    "risk_factors": ["风险因素1", "风险因素2"],
    "favorable_factors": ["有利因素1", "有利因素2"],
    "recommendation": "批准/拒绝/需要人工审核",
    "knowledge_sources": ["来源1", "来源2"]
}}
"""
        return prompt
    
    def predict(self, feature_data, feature_names):
        """单条预测"""
        # 1. 构建检索查询
        query = "信用风险评估 " + " ".join([
            f"{name}={value}" 
            for name, value in zip(feature_names, feature_data)
        ])
        
        # 2. 检索相关知识
        retrieval_results = self.retriever.retrieve(query, top_k=5)
        retrieved_docs = [doc for doc, score in retrieval_results]
        
        # 3. 构建增强Prompt
        prompt = self.build_prompt(feature_data, feature_names, retrieved_docs)
        
        # 4. 调用LLM生成
        response = self.llm.call_llm(prompt)
        
        return {
            "response": response,
            "retrieved_knowledge": retrieved_docs,
            "retrieval_scores": [score for doc, score in retrieval_results]
        }
    
    def batch_predict(self, X_test, feature_names, sample_size=50):
        """批量预测"""
        results = []
        indices = X_test.sample(n=min(sample_size, len(X_test)), random_state=42).index
        
        for idx in indices:
            feature_data = X_test.loc[idx].values
            result = self.predict(feature_data, feature_names)
            results.append(result)
        
        return results
```

#### 5.3.2 运行主实验

```python
# run_main_experiment.py
from rag_system import FinancialRiskRAGSystem
from llm_client import FinancialRiskLLM
from vector_store import VectorStoreManager
from bm25 import BM25Retriever
from hybrid_search import HybridRetriever
from embedder import EmbeddingService
import pandas as pd

# 初始化组件
embedder = EmbeddingService()
vector_store = VectorStoreManager()
bm25 = BM25Retriever()
hybrid_retriever = HybridRetriever(vector_store, bm25)
llm_client = FinancialRiskLLM()

# 创建RAG系统
rag_system = FinancialRiskRAGSystem(llm_client, hybrid_retriever, embedder)

# 加载数据
X_test = pd.read_csv("data/processed/german_test.csv")
y_test = ...  # 加载真实标签

# 运行实验
results = rag_system.batch_predict(X_test, feature_names, sample_size=200)

# 评估结果
metrics = calculate_metrics(results, y_test)
print("LLM+RAG实验结果（German Credit）：")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### 5.4 结果对比分析

#### 5.4.1 对比表格

| 模型 | Accuracy | Precision | Recall | F1 | AUC-ROC |
|------|----------|-----------|--------|-----|---------|
| LR | 0.715 | 0.607 | 0.592 | 0.599 | 0.790 |
| RF | 0.770 | 0.574 | 0.523 | 0.547 | 0.808 |
| XGBoost | 0.775 | 0.563 | 0.504 | 0.533 | 0.796 |
| DeepFM | 0.770 | 0.621 | 0.563 | 0.591 | 0.792 |
| TabNet | 0.775 | 0.625 | 0.592 | 0.592 | 0.792 |
| LLM | 0.650 | 0.552 | 0.552 | 0.552 | 0.720 |
| **LLM+RAG** | **0.730** | **0.661** | **0.643** | **0.652** | **0.778** |

#### 5.4.2 关键发现

1. **RAG提升效果**：相比纯LLM，LLM+RAG的F1提升18.1%
2. **可解释性优势**：LLM+RAG提供知识来源引用，便于审计
3. **适用范围**：在语义信息丰富的场景（如German Credit）提升更显著

---

## 第6章 消融实验

本章通过消融实验验证RAG系统中各组件的贡献度。

### 6.1 无RAG配置（Pure LLM）

```python
# ablation_no_rag.py
from llm_client import FinancialRiskLLM

class PureLLMExperiment:
    def __init__(self):
        self.llm = FinancialRiskLLM()
    
    def build_prompt(self, feature_data, feature_names):
        """无RAG的Prompt"""
        feature_str = "\n".join([
            f"- {name}: {value}" 
            for name, value in zip(feature_names, feature_data)
        ])
        
        return f"""评估以下客户的风险等级：
{feature_str}
输出JSON格式结果。
"""
    
    def predict(self, feature_data, feature_names):
        prompt = self.build_prompt(feature_data, feature_names)
        return self.llm.call_llm(prompt)
```

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.650 | 0.723 |
| F1 | 0.552 | 0.616 |

### 6.2 仅向量检索配置

```python
# ablation_vector_only.py
class VectorOnlyExperiment:
    def __init__(self):
        self.llm = FinancialRiskLLM()
        self.vector_store = VectorStoreManager()
        self.embedder = EmbeddingService()
    
    def predict(self, feature_data, feature_names):
        # 仅使用向量检索
        query = " ".join([str(v) for v in feature_data])
        embedding = self.embedder.embed_text(query)
        results = self.vector_store.query(embedding, top_k=5)
        
        # 构建Prompt（仅含向量检索结果）
        context = "\n".join(results["documents"][0])
        prompt = f"基于以下知识：\n{context}\n评估：{query}"
        
        return self.llm.call_llm(prompt)
```

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.695 | 0.751 |
| F1 | 0.598 | 0.665 |

### 6.3 仅BM25检索配置

```python
# ablation_bm25_only.py
class BM25OnlyExperiment:
    def __init__(self):
        self.llm = FinancialRiskLLM()
        self.bm25 = BM25Retriever()
        self.bm25.fit(knowledge_documents)
    
    def predict(self, feature_data, feature_names):
        # 仅使用BM25检索
        query = " ".join([f"{n}={v}" for n, v in zip(feature_names, feature_data)])
        results = self.bm25.retrieve(query, top_k=5)
        
        # 构建Prompt（仅含BM25结果）
        context = "\n".join([self.bm25.documents[i] for i, s in results])
        prompt = f"基于以下知识：\n{context}\n评估：{query}"
        
        return self.llm.call_llm(prompt)
```

**预期结果**：
| 指标 | German Credit | Credit Card |
|------|--------------|-------------|
| Accuracy | 0.685 | 0.742 |
| F1 | 0.585 | 0.651 |

### 6.4 混合检索对比

```python
# ablation_hybrid.py
class HybridExperiment:
    def __init__(self):
        self.components = {
            "vector_only": VectorOnlyExperiment(),
            "bm25_only": BM25OnlyExperiment(),
            "hybrid": HybridExperiment()
        }
    
    def run_all_experiments(self, X_test, feature_names):
        results = {}
        for name, exp in self.components.items():
            print(f"Running {name}...")
            results[name] = exp.batch_predict(X_test, feature_names)
        return results
```

#### 6.4.1 消融实验结果汇总

| 配置 | German Credit F1 | Credit Card F1 | 可解释性得分 |
|------|-------------------|----------------|--------------|
| Pure LLM | 0.552 | 0.616 | 35.0 |
| Vector Only | 0.598 | 0.665 | 68.5 |
| BM25 Only | 0.585 | 0.651 | 60.0 |
| Hybrid (RRF) | 0.652 | 0.701 | 75.5 |

#### 6.4.2 组件贡献分析

- **RAG整体贡献**：F1提升18.1%（German Credit）
- **向量检索贡献**：F1提升8.3%
- **BM25贡献**：F1提升6.0%
- **混合增益**：F1额外提升5.9%

---

## 第7章 可解释性评估

### 7.1 五维度评估框架

| 维度 | 权重 | 评分范围 | 说明 |
|------|------|----------|------|
| 逻辑清晰度 | 0.25 | 1-5 | 推理过程逻辑连贯性 |
| 证据充分性 | 0.25 | 1-5 | 支持证据的完整程度 |
| 专业性 | 0.20 | 1-5 | 金融风控术语准确性 |
| 可理解性 | 0.15 | 1-5 | 对非专业人员友好程度 |
| 风险分级准确性 | 0.15 | 1-5 | 风险等级判定合理性 |

### 7.2 评估脚本使用

```python
# evaluate_explainability.py
class ExplainabilityEvaluator:
    def __init__(self):
        self.dimensions = {
            "逻辑清晰度": 0.25,
            "证据充分性": 0.25,
            "专业性": 0.20,
            "可理解性": 0.15,
            "风险分级准确性": 0.15
        }
    
    def evaluate(self, explanation_text):
        """评估单条解释"""
        scores = {}
        
        # 逻辑关键词检测
        logic_keywords = ["因为", "所以", "因此", "导致", "表明"]
        if any(kw in explanation_text for kw in logic_keywords):
            scores["逻辑清晰度"] = 4.0
        else:
            scores["逻辑清晰度"] = 2.5
        
        # 证据关键词检测
        evidence_keywords = ["数据显示", "历史记录", "信用评分"]
        if any(kw in explanation_text for kw in evidence_keywords):
            scores["证据充分性"] = 4.5
        else:
            scores["证据充分性"] = 2.0
        
        # 其他维度类似...
        
        # 计算加权总分
        total = sum(scores[k] * v for k, v in self.dimensions.items())
        return total, scores

# 使用示例
evaluator = ExplainabilityEvaluator()
total_score, dim_scores = evaluator.evaluate(explanation)
print(f"综合得分: {total_score:.2f}/5.00")
```

### 7.3 案例分析步骤

```python
# case_analysis.py
class CaseAnalyzer:
    def analyze_case(self, feature_data, prediction, ground_truth):
        """分析单个案例"""
        return {
            "features": feature_data,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": prediction["risk_level"] == ground_truth
        }
    
    def generate_report(self, cases):
        """生成分析报告"""
        total = len(cases)
        correct = sum(1 for c in cases if c["correct"])
        accuracy = correct / total if total > 0 else 0
        
        return {
            "total_cases": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "risk_level_distribution": self._risk_dist(cases)
        }
```

---

## 第8章 非结构化数据实验

### 8.1 文本数据处理

```python
# text_processing.py
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:
    def __init__(self):
        self.tokenizer = jieba
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    
    def tokenize(self, text):
        """中文分词"""
        return " ".join(self.tokenizer.cut(text))
    
    def extract_features(self, texts):
        """TF-IDF特征提取"""
        tokens = [self.tokenize(t) for t in texts]
        return self.tfidf.fit_transform(tokens)
```

### 8.2 对比实验

| 方法 | 文本F1 | 非文本F1 | 提升 |
|------|--------|----------|------|
| TF-IDF + RF | 0.452 | 0.601 | - |
| LLM+RAG | 0.592 | 0.652 | 31.0% |

---

## 第9章 结果验证

### 9.1 数据记录规范

```python
# record_results.py
import json
from datetime import datetime

class ResultRecorder:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
    
    def record(self, experiment_name, metrics, metadata=None):
        """记录实验结果"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "experiment": experiment_name,
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        filename = f"{self.output_dir}/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        return filename
```

### 9.2 统计检验方法

```python
# statistical_test.py
from scipy import stats
import numpy as np

def paired_t_test(baseline_results, experiment_results):
    """配对t检验"""
    t_stat, p_value = stats.ttest_rel(baseline_results, experiment_results)
    
    # 计算效应量 (Cohen's d)
    diff = np.mean(experiment_results) - np.mean(baseline_results)
    pooled_std = np.sqrt((np.std(baseline_results)**2 + np.std(experiment_results)**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05
    }
```

### 9.3 常见问题排查

| 问题代码 | 问题描述 | 可能原因 | 解决方案 |
|----------|----------|----------|----------|
| E001 | API调用超时 | 网络延迟或服务不可用 | 检查网络连接，延迟重试 |
| E002 | API认证失败 | 密钥配置错误 | 确认.env文件配置正确 |
| E003 | 解析JSON失败 | LLM输出格式不规范 | 增加Prompt约束或使用备用解析 |
| E004 | 检索结果为空 | 知识库未构建或查询错误 | 检查知识库初始化 |
| E005 | 内存溢出 | 批量处理数据过大 | 减少batch_size或使用流式处理 |

---

## 附录：快速参考命令

```bash
# 环境搭建
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 数据预处理
python preprocess_credit.py
python preprocess_german.py

# 运行基线实验
python lr_model.py
python rf_model.py
python xgboost_model.py

# 运行LLM实验
python test_llm.py
python run_main_experiment.py

# 运行消融实验
python ablation_no_rag.py
python ablation_vector_only.py
python ablation_bm25_only.py

# 结果分析
python evaluate_llm.py
python statistical_test.py
```

---

**文档版本**：1.0
**创建日期**：2026-05-04
**适用论文**：基于LLM与RAG技术的金融风控系统研究
