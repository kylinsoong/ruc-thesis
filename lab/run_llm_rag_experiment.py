#!/usr/bin/env python3
import sys
import os
import csv
import time
import json
from pathlib import Path

sys.path.insert(0, '.')
from data_loader import load_data
from models.factory import create_model
from api_client import DoubaoAPIClient
from rag import BM25Retriever, VectorStoreManager, HybridRetriever, FinancialRiskRAGSystem


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


def init_rag_system(mock_mode=False):
    api_client = DoubaoAPIClient()
    kb_docs = load_knowledge_base('data/knowledge_base')
    vector_store = VectorStoreManager(persist_dir='data/knowledge_base/chroma_db')
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(kb_docs)
    embedder = EmbedderWrapper(api_client)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25_retriever, embedder=embedder)
    rag_model = create_model('llm_rag', api_client=api_client, hybrid_retriever=hybrid_retriever, embedder=embedder, mock_mode=mock_mode, random_state=42)
    return rag_model


def run_experiment(dataset, sample_size, mock_mode=False):
    print(f"Running LLM+RAG on {dataset} (mock={mock_mode}, samples={sample_size})...")

    try:
        model = init_rag_system(mock_mode=mock_mode)
        X_train, X_test, y_train, y_test = load_data(dataset)

        if len(X_test) > sample_size:
            import numpy as np
            np.random.seed(42)
            indices = np.random.choice(len(X_test), size=sample_size, replace=False)
            X_test_sampled = X_test.iloc[indices]
            y_test_sampled = y_test.iloc[indices]
        else:
            X_test_sampled = X_test
            y_test_sampled = y_test

        model.train(X_train, y_train)

        start_time = time.time()
        metrics = model.evaluate(X_test_sampled, y_test_sampled)
        elapsed = time.time() - start_time

        result = {
            'model_type': 'llm_rag',
            'dataset': dataset,
            'accuracy': f"{metrics['accuracy']:.4f}",
            'precision': f"{metrics['precision']:.4f}",
            'recall': f"{metrics['recall']:.4f}",
            'f1': f"{metrics['f1']:.4f}",
            'auc_roc': f"{metrics['auc_roc']:.4f}",
            'auc_pr': f"{metrics['auc_pr']:.4f}",
            'train_samples': len(X_train),
            'test_samples': len(X_test_sampled)
        }

        print(f"  Results: {result}")
        print(f"  Time: {elapsed:.2f}s")

        return result

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    results_file = Path('results/rag_results.csv')
    results_file.parent.mkdir(parents=True, exist_ok=True)

    experiments = [
        ('german_credit', 100),
        ('credit_card', 200),
    ]

    for dataset, sample_size in experiments:
        result = run_experiment(dataset, sample_size, mock_mode=False)

        if result is not None:
            file_exists = results_file.exists()
            with open(results_file, 'a', newline='') as f:
                fieldnames = ['model_type', 'dataset', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'train_samples', 'test_samples']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)

    print(f"\nAll results saved to {results_file}")


if __name__ == '__main__':
    main()
