import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional


class VectorStoreManager:
    def __init__(self, persist_dir="knowledge_base/chroma"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collections = ['regulations', 'industry_knowledge', 'risk_cases']
    
    def create_collection(self, name="financial_risk", dim=2048):
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"dimension": dim, "description": f"{name} collection"}
        )
        return collection

    def add_documents(self, documents, embeddings, metadatas=None, ids=None, collection_name="financial_risk"):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        collection = self.create_collection(collection_name)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def query(self, query_embedding, n_results=5, collection_name=None):
        all_results = {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "ids": [[]]
        }
        
        collections_to_query = [collection_name] if collection_name else self._collections
        
        for col_name in collections_to_query:
            try:
                collection = self.client.get_collection(name=col_name)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                if results.get("documents"):
                    all_results["documents"][0].extend(results["documents"][0])
                    all_results["distances"][0].extend(results.get("distances", [[]])[0])
                    all_results["metadatas"][0].extend(results.get("metadatas", [[]])[0])
                    all_results["ids"][0].extend(results.get("ids", [[]])[0])
            except Exception as e:
                continue
        
        if len(all_results["documents"][0]) == 0:
            return {"documents": [[]], "distances": [[]]}
        
        combined = list(zip(
            all_results["documents"][0],
            all_results["distances"][0],
            all_results["metadatas"][0],
            all_results["ids"][0]
        ))
        combined.sort(key=lambda x: x[1])
        combined = combined[:n_results]
        
        return {
            "documents": [[c[0] for c in combined]],
            "distances": [[c[1] for c in combined]],
            "metadatas": [[c[2] for c in combined]],
            "ids": [[c[3] for c in combined]]
        }

    def get_collection_info(self, collection_name=None):
        all_docs = []
        all_metadatas = []
        
        collections_to_check = [collection_name] if collection_name else self._collections
        
        for col_name in collections_to_check:
            try:
                collection = self.client.get_collection(name=col_name)
                data = collection.get()
                all_docs.extend(data.get("documents", []))
                all_metadatas.extend(data.get("metadatas", []))
            except Exception as e:
                continue
        
        return {
            "name": "all" if collection_name is None else collection_name,
            "count": len(all_docs),
            "documents": all_docs,
            "metadatas": all_metadatas
        }
    
    def get_all_documents(self):
        return self.get_collection_info().get("documents", [])
