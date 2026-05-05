import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from utils.volcengine_api import VolcEngineAPI, get_embedding
from config.config import RAG_CONFIG


class KnowledgeBase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.api = VolcEngineAPI()

    def create_collection(self, name: str = "risk_control_kb"):
        self.collection = self.client.get_or_create_collection(name=name)

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not self.collection:
            self.create_collection()

        ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        texts = [doc.get("content", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        embeddings = self.api.batch_embedding(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.collection:
            return []

        query_embedding = self.api.call_embedding(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return [
            {
                "content": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


class RAGModel:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.api = VolcEngineAPI()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.knowledge_base.query(query, top_k)

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n".join([item["content"] for item in context])
        prompt = f"""基于以下知识库内容回答问题：

知识库内容：
{context_text}

问题：{query}

请给出详细的回答："""
        return self.api.call_llm(prompt)

    def predict(self, query: str, top_k: int = 5) -> str:
        context = self.retrieve(query, top_k)
        return self.generate(query, context)
