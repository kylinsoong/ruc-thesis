import json
import os
from typing import Dict, List, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from config.config import DATA_CONFIG, MODEL_CONFIG
from utils.volcengine_api import VolcEngineAPI


class KnowledgeBase:
    def __init__(self, persist_dir: Optional[str] = None):
        self.api = VolcEngineAPI()
        self.persist_dir = persist_dir or os.path.join(DATA_CONFIG["knowledge_base_dir"], "chroma_db")
        self.embedding_model = MODEL_CONFIG["embedding"]["model_name"]
        self.embedding_dimension = MODEL_CONFIG["embedding"]["dimension"]
        
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collections: Dict[str, chromadb.Collection] = {}
    
    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[name]
    
    def load_risk_cases(self, file_path: Optional[str] = None) -> List[Dict]:
        file_path = file_path or os.path.join(DATA_CONFIG["knowledge_base_dir"], "risk_cases.json")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_regulations(self, file_path: Optional[str] = None) -> List[Dict]:
        file_path = file_path or os.path.join(DATA_CONFIG["knowledge_base_dir"], "regulations.json")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_industry_knowledge(self, file_path: Optional[str] = None) -> List[Dict]:
        file_path = file_path or os.path.join(DATA_CONFIG["knowledge_base_dir"], "industry_knowledge.json")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _create_document_text(self, data: Dict, doc_type: str) -> str:
        if doc_type == "risk_case":
            return f"""案例类型: {data['case_type']}
案例描述: {data['description']}
风险指标: {', '.join(data['risk_indicators'])}
处理决策: {data['decision']}"""
        elif doc_type == "regulation":
            return f"""来源: {data['source']}
分类: {data['category']}
规则内容: {data['content']}"""
        elif doc_type == "industry_knowledge":
            return f"""分类: {data['category']}
知识内容: {data['content']}"""
        else:
            return str(data)
    
    def _create_metadata(self, data: Dict, doc_type: str) -> Dict:
        if doc_type == "risk_case":
            return {
                "id": data["case_id"],
                "type": "risk_case",
                "case_type": data["case_type"],
                "risk_indicators": ", ".join(data["risk_indicators"])
            }
        elif doc_type == "regulation":
            return {
                "id": data["rule_id"],
                "type": "regulation",
                "source": data["source"],
                "category": data["category"]
            }
        elif doc_type == "industry_knowledge":
            return {
                "id": data["knowledge_id"],
                "type": "industry_knowledge",
                "category": data["category"]
            }
        else:
            return {"id": "", "type": "unknown"}
    
    def build_risk_cases_collection(self, data: Optional[List[Dict]] = None, batch_size: int = 10):
        if data is None:
            data = self.load_risk_cases()
        
        collection = self.get_or_create_collection("risk_cases")
        
        print(f"Building risk cases collection with {len(data)} documents...")
        
        for i in tqdm(range(0, len(data), batch_size), desc="Processing risk cases"):
            batch = data[i:i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for item in batch:
                doc_text = self._create_document_text(item, "risk_case")
                documents.append(doc_text)
                metadatas.append(self._create_metadata(item, "risk_case"))
                ids.append(item["case_id"])
            
            embeddings = self.api.batch_embedding(documents)
            
            valid_items = [
                (doc, emb, meta, id_)
                for doc, emb, meta, id_ in zip(documents, embeddings, metadatas, ids)
                if emb
            ]
            
            if valid_items:
                docs, embs, metas, ids_ = zip(*valid_items)
                collection.add(
                    documents=list(docs),
                    embeddings=list(embs),
                    metadatas=list(metas),
                    ids=list(ids_)
                )
        
        print(f"Risk cases collection built with {collection.count()} documents")
    
    def build_regulations_collection(self, data: Optional[List[Dict]] = None, batch_size: int = 10):
        if data is None:
            data = self.load_regulations()
        
        collection = self.get_or_create_collection("regulations")
        
        print(f"Building regulations collection with {len(data)} documents...")
        
        for i in tqdm(range(0, len(data), batch_size), desc="Processing regulations"):
            batch = data[i:i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for item in batch:
                doc_text = self._create_document_text(item, "regulation")
                documents.append(doc_text)
                metadatas.append(self._create_metadata(item, "regulation"))
                ids.append(item["rule_id"])
            
            embeddings = self.api.batch_embedding(documents)
            
            valid_items = [
                (doc, emb, meta, id_)
                for doc, emb, meta, id_ in zip(documents, embeddings, metadatas, ids)
                if emb
            ]
            
            if valid_items:
                docs, embs, metas, ids_ = zip(*valid_items)
                collection.add(
                    documents=list(docs),
                    embeddings=list(embs),
                    metadatas=list(metas),
                    ids=list(ids_)
                )
        
        print(f"Regulations collection built with {collection.count()} documents")
    
    def build_industry_knowledge_collection(self, data: Optional[List[Dict]] = None, batch_size: int = 10):
        if data is None:
            data = self.load_industry_knowledge()
        
        collection = self.get_or_create_collection("industry_knowledge")
        
        print(f"Building industry knowledge collection with {len(data)} documents...")
        
        for i in tqdm(range(0, len(data), batch_size), desc="Processing industry knowledge"):
            batch = data[i:i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for item in batch:
                doc_text = self._create_document_text(item, "industry_knowledge")
                documents.append(doc_text)
                metadatas.append(self._create_metadata(item, "industry_knowledge"))
                ids.append(item["knowledge_id"])
            
            embeddings = self.api.batch_embedding(documents)
            
            valid_items = [
                (doc, emb, meta, id_)
                for doc, emb, meta, id_ in zip(documents, embeddings, metadatas, ids)
                if emb
            ]
            
            if valid_items:
                docs, embs, metas, ids_ = zip(*valid_items)
                collection.add(
                    documents=list(docs),
                    embeddings=list(embs),
                    metadatas=list(metas),
                    ids=list(ids_)
                )
        
        print(f"Industry knowledge collection built with {collection.count()} documents")
    
    def build_all_collections(self, batch_size: int = 10):
        print("=" * 50)
        print("Building all knowledge base collections...")
        print("=" * 50)
        
        self.build_risk_cases_collection(batch_size=batch_size)
        self.build_regulations_collection(batch_size=batch_size)
        self.build_industry_knowledge_collection(batch_size=batch_size)
        
        print("=" * 50)
        print("All collections built successfully!")
        print("=" * 50)
    
    def search(
        self,
        query: str,
        collection_name: str = "risk_cases",
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        collection = self.get_or_create_collection(collection_name)
        
        query_embedding = self.api.call_embedding(query)
        if not query_embedding:
            print("Failed to get query embedding")
            return []
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                search_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return search_results
    
    def search_all_collections(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        results = {}
        
        for collection_name in ["risk_cases", "regulations", "industry_knowledge"]:
            try:
                collection_results = self.search(query, collection_name, top_k)
                results[collection_name] = collection_results
            except Exception as e:
                print(f"Error searching collection {collection_name}: {e}")
                results[collection_name] = []
        
        return results
    
    def search_by_case_type(
        self,
        query: str,
        case_type: str,
        top_k: int = 5
    ) -> List[Dict]:
        return self.search(
            query=query,
            collection_name="risk_cases",
            top_k=top_k,
            where={"case_type": case_type}
        )
    
    def search_by_category(
        self,
        query: str,
        category: str,
        collection_name: str = "regulations",
        top_k: int = 5
    ) -> List[Dict]:
        return self.search(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            where={"category": category}
        )
    
    def get_collection_stats(self) -> Dict[str, int]:
        stats = {}
        for collection_name in ["risk_cases", "regulations", "industry_knowledge"]:
            try:
                collection = self.get_or_create_collection(collection_name)
                stats[collection_name] = collection.count()
            except Exception:
                stats[collection_name] = 0
        return stats
    
    def delete_collection(self, name: str):
        try:
            self.client.delete_collection(name)
            if name in self.collections:
                del self.collections[name]
            print(f"Collection '{name}' deleted successfully")
        except Exception as e:
            print(f"Error deleting collection '{name}': {e}")
    
    def clear_all_collections(self):
        for name in ["risk_cases", "regulations", "industry_knowledge"]:
            self.delete_collection(name)
        print("All collections cleared")


def build_knowledge_base():
    kb = KnowledgeBase()
    kb.build_all_collections(batch_size=10)
    
    stats = kb.get_collection_stats()
    print("\nKnowledge Base Statistics:")
    for name, count in stats.items():
        print(f"  {name}: {count} documents")


if __name__ == "__main__":
    build_knowledge_base()
