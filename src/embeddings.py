import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


class EmbeddingModel:
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer = None
        
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        embeddings = []
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts")
            
        for i in iterator:
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings.append(batch_embeddings)
        
        if not embeddings:
            raise ValueError("No embeddings generated")
            
        embeddings = np.vstack(embeddings).astype("float32")

            
        return embeddings
    
    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0].astype("float32")
     
        return embedding

