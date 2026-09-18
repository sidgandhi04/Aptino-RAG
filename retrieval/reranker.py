try:
    from sentence_transformers import CrossEncoder
except OSError:
    CrossEncoder = None

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # We use a standard cross encoder for reranking
        self.model = None
        if CrossEncoder:
            try:
                self.model = CrossEncoder(model_name, max_length=512)
            except Exception as e:
                print(f"Reranker unavailable (using fused order): {e}")
        
    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
            
        if not self.model:
            # Fallback to no reranking if torch failed
            return [{"score": 1.0, **doc} for doc in documents[:top_k]]
            
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort documents by score
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        # Return top_k documents with their scores
        return [{"score": float(score), **doc} for score, doc in scored_docs[:top_k]]
