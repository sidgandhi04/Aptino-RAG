import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError) as error:
    print(f"Warning: Sentence Transformers is unavailable: {error}")
    SentenceTransformer = None


class VectorStore:
    """A small persistent cosine-similarity vector store for the policy corpus."""

    def __init__(self, db_dir="chroma_db", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_dir = db_dir
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        self._model_failed = False

        self.bm25 = None
        self.bm25_corpus = []
        self.chunk_ids = []
        self.chunk_data = {}
        self.dense_embeddings = None

        self.bm25_path = os.path.join(self.db_dir, "bm25_index.pkl")
        self.dense_path = os.path.join(self.db_dir, "dense_index.pkl")
        self._load_bm25()
        self._load_dense()

    @property
    def embedding_model(self):
        if self._embedding_model is None and not self._model_failed and SentenceTransformer:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as error:
                print(f"Warning: Dense embedding model is unavailable: {error}")
                self._model_failed = True
        return self._embedding_model

    def _load_bm25(self):
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as file:
                data = pickle.load(file)
            self.bm25 = data["bm25"]
            self.bm25_corpus = data["corpus"]
            self.chunk_ids = data["chunk_ids"]
            self.chunk_data = data.get("chunk_data", {})

    def _load_dense(self):
        if os.path.exists(self.dense_path):
            with open(self.dense_path, "rb") as file:
                data = pickle.load(file)
            self.dense_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            if not self.chunk_ids:
                self.chunk_ids = data["chunk_ids"]
            if not self.chunk_data:
                self.chunk_data = data["chunk_data"]

    def _save_indexes(self):
        os.makedirs(self.db_dir, exist_ok=True)
        with open(self.bm25_path, "wb") as file:
            pickle.dump({
                "bm25": self.bm25,
                "corpus": self.bm25_corpus,
                "chunk_ids": self.chunk_ids,
                "chunk_data": self.chunk_data,
            }, file)
        with open(self.dense_path, "wb") as file:
            pickle.dump({
                "embeddings": self.dense_embeddings,
                "chunk_ids": self.chunk_ids,
                "chunk_data": self.chunk_data,
            }, file)

    def add_chunks(self, chunks):
        if not self.embedding_model:
            raise RuntimeError(
                "Dense embedding model is unavailable."
            )

        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.chunk_data = {
            chunk["chunk_id"]: {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }
            for chunk in chunks
        }
        documents = [chunk["text"] for chunk in chunks]
        self.bm25_corpus = [text.lower().split() for text in documents]

        print(f"Computing embeddings for {len(chunks)} chunks...")
        self.dense_embeddings = np.asarray(
            self.embedding_model.encode(documents, normalize_embeddings=True),
            dtype=np.float32,
        )
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.bm25_corpus)
        self._save_indexes()
        print("Indexing complete.")

    def dense_search(self, query, top_k):
        if not self.embedding_model or self.dense_embeddings is None:
            return []
        if len(self.dense_embeddings) != len(self.chunk_ids):
            raise RuntimeError("Dense index and chunk metadata are out of sync.")

        query_embedding = np.asarray(
            self.embedding_model.encode([query], normalize_embeddings=True)[0],
            dtype=np.float32,
        )
        scores = self.dense_embeddings @ query_embedding
        top_indices = np.argsort(-scores)[:min(top_k, len(scores))]
        return [
            {
                "chunk_id": self.chunk_ids[index],
                "text": self.chunk_data[self.chunk_ids[index]]["text"],
                "metadata": self.chunk_data[self.chunk_ids[index]]["metadata"],
            }
            for index in top_indices
        ]
