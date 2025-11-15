from abc import ABC, abstractmethod
from typing import List
import openai

class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Embedding model using OpenAI's API."""
    def __init__(self, client: openai.OpenAI, model_name: str = "text-embedding-3-small"):
        self.client = client
        self.model_name = model_name
        # Known dimensions for OpenAI models
        self._dimensions = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
        if model_name not in self._dimensions:
            raise ValueError(f"Unknown dimension for OpenAI model: {model_name}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in response.data]

    @property
    def dimension(self) -> int:
        return self._dimensions[self.model_name]

class LocalEmbeddingModel(BaseEmbeddingModel):
    """Embedding model using a local sentence-transformer."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install 'sentence-transformers' to use local embedding models: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        print(f"Initialized local embedding model '{model_name}' with dimension {self._dimension}.")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # The model can return numpy arrays, so we convert them to lists
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension