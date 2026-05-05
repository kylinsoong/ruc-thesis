# LLM模型实验运行分析文档

## 命令
```bash
python run_baseline_experiment.py --model llm --dataset german_credit
```

---

## 一、数据集详情

### German Credit数据集

| 属性 | 值 |
|------|-----|
| 训练集大小 | 700条 |
| 测试集大小 | 200条 |
| 特征数量 | 20个 |
| 类别 | 二分类（好信用/坏信用）|

### 测试集类别分布
- 好信用（0）：约70%
- 坏信用（1）：约30%

### 特征列表

| 序号 | 特征名 | 说明 |
|------|--------|------|
| 1 | checking_status | 活期账户状态 |
| 2 | duration | 贷款期限（月） |
| 3 | credit_history | 信用记录 |
| 4 | purpose | 贷款目的 |
| 5 | credit_amount | 贷款金额 |
| 6 | savings_status | 储蓄账户状态 |
| 7 | employment | 就业年限 |
| 8 | installment_commitment | 分期付款比例 |
| 9 | personal_status | 性别/婚姻状态 |
| 10 | other_parties | 其他担保人 |
| 11 | residence_since | 居住时长 |
| 12 | property_magnitude | 财产状况 |
| 13 | age | 年龄 |
| 14 | other_payment_plans | 其他分期计划 |
| 15 | housing | 住房状况 |
| 16 | existing_credits | 现有贷款数 |
| 17 | job | 工作类型 |
| 18 | num_dependents | 家属数量 |
| 19 | own_telephone | 是否有电话 |
| 20 | foreign_worker | 是否外籍工人 |

---

## 二、整体运行过程

### 入口流程

```
main() [run_baseline_experiment.py:161]
├── 解析命令行参数
│   └── model="llm", dataset="german_credit"
├── 调用 run_model_experiment("llm", "german_credit")
│   └── 执行完整实验流程
└── 保存结果到CSV
```

### 详细步骤

| 步骤 | 函数/位置 | 操作 |
|------|----------|------|
| 1 | main() | 解析参数：model=llm, dataset=german_credit |
| 2 | main() | 确定输出路径：results/baseline_results.csv |
| 3 | run_model_experiment() | 调用 load_data("german_credit") |
| 4 | run_model_experiment() | 打印数据集信息 |
| 5 | run_model_experiment() | 创建LLM模型实例 |
| 6 | run_model_experiment() | 调用 model.evaluate(X_test, y_test) |
| 7 | run_model_experiment() | 保存结果到CSV |
| 8 | main() | 打印完成信息 |

---

## 三、单次运行详细过程

### 3.1 数据加载阶段

```python
# run_baseline_experiment.py:66
X_train, X_test, y_train, y_test = load_data("german_credit")
```

**数据来源**：`lab/data/processed/`
- german_X_train.csv（700条）
- german_X_test.csv（200条）
- german_y_train.csv
- german_y_test.csv

**耗时**：< 1秒

---

### 3.2 模型创建阶段

```python
# run_baseline_experiment.py:78
model = create_model("llm", mock_mode=False, random_state=42)
```

**创建LLMOnlyModel实例**：
- 初始化DoubaoAPIClient
- 设置temperature=0.3
- 设置max_retries=3
- 设置mock_mode=False

**耗时**：< 1秒

---

### 3.3 评估阶段（核心耗时）

```python
# run_baseline_experiment.py:92
metrics = model.evaluate(X_test, y_test)
```

#### evaluate()执行流程

```python
# llm_only.py:150-174
def evaluate(X_test, y_test):
    1. y_pred = self.predict(X_test)      # API调用 × 200
    2. y_prob = self.predict_proba(X_test) # API调用 × 200
    3. 计算metrics
```

#### predict()执行流程

```python
# llm_only.py:112-129
def predict(X_test):
    for idx, row in X_test.iterrows():  # 200次循环
        _, label = self.predict_single(row.to_dict())
    return predictions
```

#### predict_proba()执行流程

```python
# llm_only.py:131-148
def predict_proba(X_test):
    for idx, row in X_test.iterrows():  # 200次循环
        risk_score, _ = self.predict_single(row.to_dict())
    return probabilities
```

#### predict_single()执行流程

```python
# llm_only.py:80-110
def predict_single(features):
    1. 构建prompt: _build_prompt()       # < 1秒
    2. 调用API: api_client.call_llm()    # ~12秒
    3. 解析响应: _parse_response()       # < 1秒
    4. 失败重试: 最多3次                   # +0.5~1秒/次
    5. 预测间隔: time.sleep(0.5)          # 0.5秒
```

---

### 3.4 API调用详情

```python
# api_client.py:21-40
def call_llm(prompt, temperature=0.3):
    response = client.responses.create(
        model="doubao-seed-2-0-pro-260215",
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        temperature=0.3
    )
    return {"content": response.output_text}
```

**API信息**：
| 属性 | 值 |
|------|-----|
| 模型 | doubao-seed-2-0-pro-260215 |
| 端点 | https://ark.cn-beijing.volces.com/api/v3 |
| Temperature | 0.3 |

---

### 3.5 Prompt模板

```python
# llm_only.py:15-31
RISK_ANALYSIS_PROMPT = """你是一位资深的金融风控专家。请根据以下客户特征数据进行风险评估。

## 客户特征
{feature_str}

## 评估要求
1. 分析各特征的风险信号
2. 给出风险评分（0-1之间，1表示高风险）
3. 给出审批建议（批准/拒绝）

## 输出格式
请严格按照以下JSON格式输出：
{
    "risk_score": 0.0到1.0之间的风险评分,
    "recommendation": "批准"或"拒绝"
}
"""
```

**示例Prompt（当feature_str为"- checking_status: 1"时）**：
```
你是一位资深的金融风控专家。请根据以下客户特征数据进行风险评估。

## 客户特征
- checking_status: 1
- duration: -0.2408572253028645
...

## 评估要求
1. 分析各特征的风险信号
2. 给出风险评分（0-1之间，1表示高风险）
3. 给出审批建议（批准/拒绝）

## 输出格式
请严格按照以下JSON格式输出：
{
    "risk_score": 0.0到1.0之间的风险评分,
    "recommendation": "批准"或"拒绝"
}
```

---

### 3.6 响应解析

```python
# llm_only.py:58-72
def _parse_response(response_text):
    1. 正则提取JSON: r'\{[^}]+\}'
    2. 解析JSON获取 risk_score 和 recommendation
    3. 转换: "批准"→0, "拒绝"→1
```

**期望响应格式**：
```json
{
    "risk_score": 0.7,
    "recommendation": "拒绝"
}
```

---

### 3.7 单样本耗时分解

| 步骤 | 代码位置 | 耗时 |
|------|---------|------|
| Prompt构建 | llm_only.py:92 | < 0.1秒 |
| API调用 | api_client.py:23 | ~12秒 |
| 响应解析 | llm_only.py:100 | < 0.1秒 |
| 预测间隔 | llm_only.py:104 | 0.5秒 |
| **单样本总计** | | **~12.5秒** |

---

## 四、完整耗时计算

### API调用次数

| 方法 | 样本数 | API调用次数 |
|------|--------|------------|
| predict() | 200 | 200次 |
| predict_proba() | 200 | 200次 |
| **总计** | | **400次** |

### 耗时估算

```
单样本API耗时：~12秒
总样本数：200条

predict()耗时：200 × 12.5 = 2500秒
predict_proba()耗时：200 × 12.5 = 2500秒

总耗时 ≈ 5000秒 ≈ 83分钟 ≈ 1小时23分钟
```

### 实际单次测试结果

| 测试 | 耗时 |
|------|------|
| 单样本（包含predict + predict_proba） | **48秒** |

### 预估总耗时

```
German Credit (200条)：
- 最低：48秒 × 200 / 4 = 2400秒 ≈ 40分钟
- 最高：48秒 × 200 = 9600秒 ≈ 160分钟
- 预估：约1.5~2.5小时
```

---

## 五、结果输出

### 输出格式

```csv
model_type,dataset,accuracy,precision,recall,f1,auc_roc,auc_pr,train_samples,test_samples
llm,german_credit,0.XXXX,0.XXXX,0.XXXX,0.XXXX,0.XXXX,0.XXXX,700,200
```

### 输出字段说明

| 字段 | 说明 |
|------|------|
| model_type | 模型类型（llm） |
| dataset | 数据集名称（german_credit） |
| accuracy | 准确率 |
| precision | 精确率 |
| recall | 召回率 |
| f1 | F1分数 |
| auc_roc | ROC曲线下面积 |
| auc_pr | PR曲线下面积 |
| train_samples | 训练样本数 |
| test_samples | 测试样本数 |

---

## 五-2、评估指标计算详解

### 5.1 混淆矩阵基础

根据 [llm_only.py:150-174](file:///Users/bytedance/src/coding.ai/ruc/lab/models/baseline/llm_only.py#L150-L174)，评估指标计算如下：

```python
y_pred = self.predict(X_test)      # 预测标签（0或1）
y_prob = self.predict_proba(X_test) # 预测概率（0~1之间的风险评分）
```

**German Credit数据集约定**：
- 标签0：好信用（批准）
- 标签1：坏信用（拒绝）

### 5.2 混淆矩阵

```
                    预测结果
                ┌─────────┬─────────┐
                │   0     │   1     │
    ┌───────┬─────────┼─────────┤
    │   0   │   TN    │   FP    │
    │ 真实  ├─────────┼─────────┤
    │   1   │   FN    │   TP    │
    └───────┴─────────┴─────────┘

TN (True Negative)  ：真实为好信用，预测为批准
FP (False Positive) ：真实为好信用，预测为拒绝
FN (False Negative) ：真实为坏信用，预测为批准
TP (True Positive)  ：真实为坏信用，预测为拒绝
```

### 5.3 各指标计算公式

#### Accuracy（准确率）
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**含义**：预测正确的样本占总样本的比例

**计算示例**：
- 假设200条测试集中，150条预测正确
- Accuracy = 150/200 = 0.75

#### Precision（精确率）
```python
precision = TP / (TP + FP)
```
**含义**：预测为坏信用的样本中，实际为坏信用的比例

**计算示例**：
- 假设预测拒绝60条（TP+FP=60），其中实际坏信用45条（TP=45）
- Precision = 45/60 = 0.75

#### Recall（召回率）
```python
recall = TP / (TP + FN)
```
**含义**：实际为坏信用的样本中，被正确预测为拒绝的比例

**计算示例**：
- 假设实际坏信用50条（TP+FN=50），其中被正确识别45条（TP=45）
- Recall = 45/50 = 0.90

#### F1 Score（F1分数）
```python
f1 = 2 * (precision * recall) / (precision + recall)
```
**含义**：精确率和召回率的调和平均

**计算示例**：
- Precision=0.75, Recall=0.90
- F1 = 2 × (0.75 × 0.90) / (0.75 + 0.90) = 1.35 / 1.65 = 0.818

### 5.4 AUC指标计算

#### AUC-ROC（ROC曲线下面积）
```python
metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob))
```
**含义**：根据预测概率 `y_prob`（风险评分）与真实标签 `y_test` 计算ROC曲线下面积

**计算方法**：
- 将预测概率从高到低排序
- 依次将每个值作为阈值，计算真阳性率和假阳性率
- 绘制ROC曲线，计算曲线下面积

**解读**：
- AUC = 1.0：完美模型
- AUC = 0.5：随机猜测
- AUC > 0.7：较好模型

#### AUC-PR（PR曲线下面积）
```python
metrics["auc_pr"] = float(average_precision_score(y_test, y_prob))
```
**含义**：PR曲线的下面积，适用于类别不平衡情况

**特点**：
- German Credit数据集类别比例约7:3（相对平衡）
- 在不平衡数据集中比AUC-ROC更可靠

### 5.5 predict_single返回值含义

```python
# llm_only.py:64-68
risk_score = float(result.get("risk_score", 0.5))  # 0.0~1.0，1表示高风险
recommendation = result.get("recommendation", "批准")  # "批准"或"拒绝"

label = 0 if recommendation == "批准" else 1  # 转换为0/1标签
```

| LLM返回 | risk_score | recommendation | 转换后label |
|---------|-----------|----------------|-------------|
| 批准低风险 | 0.3 | "批准" | 0 |
| 拒绝高风险 | 0.8 | "拒绝" | 1 |

### 5.6 完整计算示例

假设测试集200条，其中70条坏信用（y=1），130条好信用（y=0）

**模型预测结果**：
| 类别 | 预测为0 | 预测为1 | 合计 |
|------|---------|---------|------|
| 真实0 | TN=110 | FP=20 | 130 |
| 真实1 | FN=15 | TP=55 | 70 |
| 合计 | 125 | 75 | 200 |

**计算指标**：
| 指标 | 计算 | 结果 |
|------|------|------|
| Accuracy | (110+55)/200 | 0.825 |
| Precision | 55/(55+20) | 0.733 |
| Recall | 55/(55+15) | 0.786 |
| F1 | 2×0.733×0.786/(0.733+0.786) | 0.759 |

---

## 六、流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_baseline_experiment.py                   │
├─────────────────────────────────────────────────────────────────┤
│  main()                                                          │
│  ├── 解析参数: --model llm --dataset german_credit              │
│  └── run_model_experiment("llm", "german_credit")               │
│                                                                  │
│  run_model_experiment()                                          │
│  ├── load_data("german_credit")  [200条测试数据]                  │
│  ├── create_model("llm")                                         │
│  │                                                                │
│  │   ┌─────────────────────────────────────────────────────┐    │
│  │   │              model.evaluate(X_test, y_test)          │    │
│  │   ├─────────────────────────────────────────────────────┤    │
│  │   │  predict(X_test)                                    │    │
│  │   │  ├── for each sample (200次)                        │    │
│  │   │   │   predict_single() → API调用(~12秒)            │    │
│  │   │   └── time.sleep(0.5)                              │    │
│  │   │  返回: y_pred (200个标签)                           │    │
│  │   ├─────────────────────────────────────────────────────┤    │
│  │   │  predict_proba(X_test)                              │    │
│  │   │  ├── for each sample (200次)                        │    │
│  │   │   │   predict_single() → API调用(~12秒)            │    │
│  │   │   └── time.sleep(0.5)                              │    │
│  │   │  返回: y_prob (200个概率)                           │    │
│  │   ├─────────────────────────────────────────────────────┤    │
│  │   │  计算指标: accuracy, precision, recall, f1, auc   │    │
│  │   └─────────────────────────────────────────────────────┘    │
│  │                                                                │
│  └── save_result() → CSV                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、关键代码文件

| 文件 | 作用 |
|------|------|
| [run_baseline_experiment.py](file:///Users/bytedance/src/coding.ai/ruc/lab/run_baseline_experiment.py) | 实验入口，主流程控制 |
| [llm_only.py](file:///Users/bytedance/src/coding.ai/ruc/lab/models/baseline/llm_only.py) | LLM模型实现，API调用 |
| [api_client.py](file:///Users/bytedance/src/coding.ai/ruc/lab/api_client.py) | 豆包API客户端 |
| [data_loader.py](file:///Users/bytedance/src/coding.ai/ruc/lab/data_loader.py) | 数据加载 |
| [factory.py](file:///Users/bytedance/src/coding.ai/ruc/lab/models/factory.py) | 模型工厂 |

---

## 八、预估运行时间汇总

| 阶段 | 耗时 |
|------|------|
| 数据加载 | < 1秒 |
| 模型创建 | < 1秒 |
| predict() 200条 | ~20分钟 |
| predict_proba() 200条 | ~20分钟 |
| 结果保存 | < 1秒 |
| **总计** | **约40~50分钟** |
