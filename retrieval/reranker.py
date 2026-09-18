try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

try:
    from sentence_transformers import CrossEncoder
except (ImportError, OSError):
    CrossEncoder = None


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._failed = False

    @property
    def model(self):
        if self._model is None and not self._failed and CrossEncoder:
            try:
                self._model = CrossEncoder(self.model_name, max_length=512)
            except Exception as e:
                print(f"Reranker unavailable (using fused order): {e}")
                self._failed = True
        return self._model

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []

        if not self.model:
            # Fallback to RRF order
            return [{"score": 1.0, **doc} for doc in documents[:top_k]]

        pairs = [[query, doc["text"]] for doc in documents]
        try:
            scores = self.model.predict(pairs)
            scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [{"score": float(score), **doc} for score, doc in scored_docs[:top_k]]
        except Exception as error:
            print(f"Reranking prediction failed: {error}")
            return [{"score": 1.0, **doc} for doc in documents[:top_k]]
