#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from data_loader import load_data
from models.factory import create_model
from api_client import DoubaoAPIClient
from rag import BM25Retriever, VectorStoreManager, HybridRetriever
import json
from pathlib import Path
import numpy as np

def load_knowledge_base(kb_dir):
    kb_path = Path(kb_dir)
    all_docs = []
    json_files = ['industry_knowledge.json', 'regulations.json', 'risk_cases.json']
    for fname in json_files:
        fpath = kb_path / fname
        if fpath.exists():
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        content = item.get('content', '') or item.get('description', '') or item.get('case_summary', '')
                        if content:
                            all_docs.append(content)
    return all_docs

class EmbedderWrapper:
    def __init__(self, api_client):
        self.api_client = api_client
    def embed_text(self, text):
        result = self.api_client.get_embedding(text)
        if 'response' in result:
            resp = result['response']
            if hasattr(resp, 'embedding'):
                return resp.embedding
            elif hasattr(resp, 'data') and len(resp.data) > 0:
                return resp.data[0].embedding
        return [0.0] * 2048

if __name__ == '__main__':
    print('Initializing RAG system...')
    api_client = DoubaoAPIClient()
    kb_docs = load_knowledge_base('data/knowledge_base')
    print(f'Loaded {len(kb_docs)} knowledge base documents')

    vector_store = VectorStoreManager(persist_dir='data/knowledge_base/chroma_db')
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(kb_docs)
    print('BM25 index built')

    embedder = EmbedderWrapper(api_client)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25_retriever, embedder=embedder)

    X_train, X_test, y_train, y_test = load_data('german_credit')
    print(f'Data loaded: test size = {len(X_test)}')

    print('Testing hybrid retrieval...')
    test_query = '信用风险评估 贷款 负债'
    results = hybrid_retriever.hybrid_retrieve(test_query, top_k=3)
    print(f'Retrieval results: {len(results)} docs found')
    for doc, score in results[:2]:
        print(f'  Score {score:.4f}: {doc[:80]}...')

    print('\nCreating RAG model...')
    rag_model = create_model('llm_rag', api_client=api_client, hybrid_retriever=hybrid_retriever, embedder=embedder, mock_mode=False)
    rag_model.train(X_train, y_train)

    print('Running evaluation on small sample (20 samples)...')
    np.random.seed(42)
    indices = np.random.choice(len(X_test), size=20, replace=False)
    X_test_sample = X_test.iloc[indices]
    y_test_sample = y_test.iloc[indices]

    metrics = rag_model.evaluate(X_test_sample, y_test_sample)
    print(f'\nResults (20 samples):')
    print(f'  Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall:    {metrics["recall"]:.4f}')
    print(f'  F1:        {metrics["f1"]:.4f}')
    print(f'  AUC-ROC:   {metrics["auc_roc"]:.4f}')
    print(f'  AUC-PR:    {metrics["auc_pr"]:.4f}')
