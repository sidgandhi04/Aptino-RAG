from ingestion.vector_store import VectorStore
from retrieval.reranker import Reranker

class HybridSearcher:
    def __init__(self, db_dir="chroma_db", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.store = VectorStore(db_dir=db_dir, embedding_model_name=embedding_model_name)
        self.reranker = Reranker()
        
    def _rrf(self, dense_results, sparse_results, k=60):
        # Reciprocal Rank Fusion
        scores = {}
        
        # Dense results processing
        for rank, doc in enumerate(dense_results):
            chunk_id = doc["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = {"score": 0.0, "doc": doc}
            scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
            
        # Sparse results processing
        for rank, doc in enumerate(sparse_results):
            chunk_id = doc["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = {"score": 0.0, "doc": doc}
            scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
            
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def search(self, query, top_k=5, pre_rerank_k=15):
        # 1. Dense search
        dense_docs = []
        if self.store.embedding_model:
            try:
                dense_docs = self.store.dense_search(query, pre_rerank_k)
            except Exception as e:
                print(f"Dense search failed (falling back to BM25 only): {e}")
                
        # 2. Sparse Search (BM25)
        sparse_docs = []
        if self.store.bm25:
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self.store.bm25.get_scores(tokenized_query)
                top_n = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:pre_rerank_k]

                for idx in top_n:
                    if bm25_scores[idx] <= 0:
                        continue
                    chunk_id = self.store.chunk_ids[idx]
                    chunk_data = self.store.chunk_data.get(chunk_id)
                    if chunk_data:
                        sparse_docs.append({
                            "chunk_id": chunk_id,
                            "text": chunk_data["text"],
                            "metadata": chunk_data["metadata"],
                        })
                    else:
                        # An old BM25 pickle without chunk metadata must not
                        # make lexical retrieval depend on (or crash with) the
                        # vector store. Rebuild the index instead.
                        print(
                            f"BM25 metadata missing for {chunk_id}; "
                            "skipping stale lexical result."
                        )
            except Exception as e:
                print(f"Sparse search failed: {e}")
                        
        # 3. Fuse results
        fused_docs = self._rrf(dense_docs, sparse_docs)
        
        # 4. Rerank
        try:
            reranked_docs = self.reranker.rerank(
                query, fused_docs[:pre_rerank_k * 2], top_k=top_k
            )
        except Exception as e:
            print(f"Reranking failed (using fused order): {e}")
            reranked_docs = [
                {"score": doc_score["score"], **doc_score["doc"]}
                for doc_score in sorted(
                    self._rrf_scored(dense_docs, sparse_docs),
                    key=lambda item: item["score"],
                    reverse=True,
                )[:top_k]
            ]
        
        return reranked_docs

    def _rrf_scored(self, dense_results, sparse_results, k=60):
        """Return RRF scores as well as documents for a reranker fallback."""
        scores = {}
        for results in (dense_results, sparse_results):
            for rank, doc in enumerate(results):
                item = scores.setdefault(doc["chunk_id"], {"score": 0.0, "doc": doc})
                item["score"] += 1.0 / (k + rank + 1)
        return list(scores.values())

if __name__ == "__main__":
    searcher = HybridSearcher()
    res = searcher.search("cataract waiting period")
    for r in res:
        print(f"[{r['score']}] {r['chunk_id']}: {r['text'][:100]}...")
