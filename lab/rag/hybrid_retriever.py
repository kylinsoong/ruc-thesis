from typing import List, Tuple, Dict, Any


def reciprocal_rank_fusion(results_list: List[List[Tuple[Any, float]]], k: int = 60) -> List[Tuple[Any, float]]:
    scores: Dict[Any, float] = {}
    
    for results in results_list:
        for rank, (doc_id, score) in enumerate(results, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1.0 / (k + rank) + score * 0.1
    
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, embedder, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.embedder = embedder
        self.rrf_k = rrf_k
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        vector_results = []
        try:
            query_embedding = self.embedder.embed_text(query)
            if isinstance(query_embedding, list) and len(query_embedding) > 0:
                query_embedding_list = query_embedding
            else:
                query_embedding_list = [0.0] * 2048
            
            vector_results_raw = self.vector_store.query(query_embedding_list, n_results=top_k * 2)
            
            if vector_results_raw and "documents" in vector_results_raw:
                vector_docs = vector_results_raw["documents"][0]
                vector_distances = vector_results_raw.get("distances", [[]])[0]
                for doc, dist in zip(vector_docs, vector_distances):
                    similarity = 1.0 / (1.0 + dist) if dist else 0.0
                    vector_results.append((doc, similarity))
        except Exception as e:
            vector_results = []
        
        bm25_results = []
        try:
            all_docs = self.vector_store.get_all_documents()
            if len(all_docs) == 0:
                all_docs = self.bm25.documents
            
            bm25_raw = self.bm25.retrieve(query, top_k * 2)
            for idx, score in bm25_raw:
                if isinstance(idx, int) and idx < len(all_docs):
                    doc = all_docs[idx]
                    bm25_results.append((doc, score))
                elif isinstance(idx, str):
                    bm25_results.append((idx, score))
        except Exception as e:
            bm25_results = []
        
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.rrf_k
        )
        
        return fused_results[:top_k]
