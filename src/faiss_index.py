import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

from src.config import (
    FAISS_INDEX_PATH,
    FAISS_IDS_PATH,
    FAISS_META_PATH,
    DOCUMENTS_JSONL_PATH,
    DEFAULT_TOP_K,
)
from src.embeddings import EmbeddingModel
from src.data_processing import load_documents


class FAISSIndex:
    
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        ids_path: Path = FAISS_IDS_PATH,
        meta_path: Path = FAISS_META_PATH
    ):
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.meta_path = Path(meta_path)
        
        self.index: faiss.Index = None
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
    def build(
        self,
        documents_path: Path = DOCUMENTS_JSONL_PATH,
        embedding_model: EmbeddingModel = None
    ) -> None:
        print("Building FAISS index...")
        
        print(f"Loading documents from: {documents_path}")
        documents = load_documents(documents_path)
        if not documents:
            raise ValueError(f"No documents found in {documents_path}")
        print(f"Loaded {len(documents)} documents")
        
        print("Preparing items for indexing...")
        ids, texts, metadata = self._prepare_items(documents)
        print(f"Items to index: {len(ids)}")
        
        if embedding_model is None:
            embedding_model = EmbeddingModel()
        
        print("Computing embeddings...")
        embeddings = embedding_model.encode_texts(
            texts,
            show_progress=True,
        )
        
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.ids = ids
        self.metadata = metadata
        
        print(f"Index built with {self.index.ntotal} vectors.")
        
    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("No index to save. Build or load an index first.")
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving FAISS index to: {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))
        
        print(f"Saving IDs to: {self.ids_path}")
        np.save(str(self.ids_path), np.array(self.ids, dtype=object))
        
        print(f"Saving metadata to: {self.meta_path}")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for meta in self.metadata:
                json.dump(meta, f, ensure_ascii=False)
                f.write("\n")
        
        print("Index saved successfully.")
        
    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Build the index first using build_index.py"
            )
        
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        print(f"Loading IDs from: {self.ids_path}")
        self.ids = np.load(str(self.ids_path), allow_pickle=True).tolist()
        
        print(f"Loading metadata from: {self.meta_path}")
        self.metadata = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        print(f"Loaded index with {len(self.ids)} items.")
        
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("No index loaded. Call load() first.")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k
        )
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            
            results.append({
                "id": self.ids[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def query(
        self,
        query_text: str,
        embedding_model: EmbeddingModel = None,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        if embedding_model is None:
            embedding_model = EmbeddingModel()
        
        query_embedding = embedding_model.encode_query(
            query_text,
        )
        
        return self.search(query_embedding, top_k=top_k)
    
    @staticmethod
    def _prepare_items(
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        ids, texts, metadata = [], [], []
        
        for doc in documents:
            doc_id = doc.get("id", "").strip()
            text = str(doc.get("text", "")).strip()
            meta = doc.get("metadata", {}) or {}
            
            if not doc_id or not text:
                continue
            
            ids.append(doc_id)
            texts.append(text)
            metadata.append(meta)
        
        return ids, texts, metadata


def search_tickets(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    faiss_index: FAISSIndex = None
) -> List[Dict[str, Any]]:
    if faiss_index is None:
        faiss_index = FAISSIndex()
        faiss_index.load()
    
    return faiss_index.query(query, top_k=top_k)

