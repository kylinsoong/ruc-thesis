class FinancialRiskRAGSystem:
    def __init__(self, llm_client, hybrid_retriever, embedder):
        self.llm = llm_client
        self.retriever = hybrid_retriever
        self.embedder = embedder

    def build_prompt(self, feature_data, feature_names, retrieved_knowledge):
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
        query = "信用风险评估 " + " ".join([
            f"{name}={value}" 
            for name, value in zip(feature_names, feature_data)
        ])
        
        retrieval_results = self.retriever.hybrid_retrieve(query, top_k=5)
        
        retrieved_docs = [doc for doc, score in retrieval_results]
        
        prompt = self.build_prompt(feature_data, feature_names, retrieved_docs)
        response = self.llm.call_llm(prompt)
        
        return {
            "response": response,
            "retrieved_knowledge": retrieved_docs,
            "retrieval_scores": [score for doc, score in retrieval_results]
        }

    def batch_predict(self, X_test, feature_names, sample_size=50):
        results = []
        indices = X_test.sample(n=min(sample_size, len(X_test)), random_state=42).index
        
        for idx in indices:
            feature_data = X_test.loc[idx].values
            result = self.predict(feature_data, feature_names)
            results.append(result)
        
        return results
