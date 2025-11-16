from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
import torch

# Import transformers at module level to avoid lazy import overhead
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None

# Global cache for embedding models to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}
_LOADING_MARKER = "_LOADING_"  # Marker to prevent duplicate loading


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


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling - Take attention mask into account for correct averaging.

    Accept either the tuple-style output (model_output[0]) or an object with
    `.last_hidden_state` (transformers BaseModelOutput).
    """
    if isinstance(model_output, tuple):
        token_embeddings = model_output[0]
    elif hasattr(model_output, "last_hidden_state"):
        token_embeddings = model_output.last_hidden_state
    else:
        # fallback - try indexing
        token_embeddings = model_output[0]

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, 1)
    counts = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return summed / counts


class LocalEmbeddingModel(BaseEmbeddingModel):
    """Embedding model using raw transformers (faster than sentence-transformers).

    Changes made:
    - Uses the fast Rust tokenizer (use_fast=True).
    - Avoids a dummy forward pass to infer embedding dimension; reads from model.config.hidden_size.
      Only performs a forward pass as a rare fallback if hidden_size isn't present.
    - Supports optional cache_dir for pre-populated local cache.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
                 cache_dir: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install 'transformers' to use local embedding models: pip install transformers torch")

        self.model_name = model_name
        self.cache_dir = cache_dir

        # Reuse already-loaded model if present and fully loaded
        if model_name in _MODEL_CACHE and _MODEL_CACHE[model_name] != _LOADING_MARKER:
            print(f"Reusing cached embedding model '{model_name}'.")
            self.tokenizer, self.model = _MODEL_CACHE[model_name]
            # Get dimension from model config (very fast, no forward pass)
            self._dimension = getattr(self.model.config, "hidden_size", None)
            if self._dimension is None:
                # Rare fallback: perform a tiny forward pass to determine dimension
                print("Warning: model.config.hidden_size missing — performing one tiny forward pass as fallback.")
                with torch.no_grad():
                    dummy = self.tokenizer(["test"], padding=True, truncation=True, return_tensors='pt')
                    output = self.model(**dummy)
                    self._dimension = mean_pooling(output, dummy['attention_mask']).shape[1]
            return

        # If another thread/process is currently loading, wait for it to finish
        if model_name in _MODEL_CACHE and _MODEL_CACHE[model_name] == _LOADING_MARKER:
            import time
            print(f"Waiting for embedding model '{model_name}' to finish loading...")
            while _MODEL_CACHE.get(model_name) == _LOADING_MARKER:
                time.sleep(0.05)
            self.tokenizer, self.model = _MODEL_CACHE[model_name]
            self._dimension = getattr(self.model.config, "hidden_size", None) or 0
            return

        # Otherwise, load and cache the model (marking so concurrent inits wait)
        _MODEL_CACHE[model_name] = _LOADING_MARKER
        print(f"Initializing local embedding model '{model_name}'...")

        # Use the Rust-based fast tokenizer for much faster tokenization startup & runtime.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)

        # Try to prefer safetensors where available (faster & safer), but fall back if not supported
        try:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, use_safetensors=True)
        except TypeError:
            # some older transformers versions don't accept use_safetensors
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # Get dimension from model config — avoids doing a forward pass in the normal case
        self._dimension = getattr(self.model.config, "hidden_size", None)
        if self._dimension is None:
            # last-resort fallback (should be very rare)
            with torch.no_grad():
                dummy = self.tokenizer(["test"], padding=True, truncation=True, return_tensors='pt')
                output = self.model(**dummy)
                self._dimension = mean_pooling(output, dummy['attention_mask']).shape[1]

        # store into cache for other instances
        _MODEL_CACHE[model_name] = (self.tokenizer, self.model)
        print(f"Initialized local embedding model '{model_name}' with dimension {self._dimension}.")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Tokenize and encode using the fast tokenizer
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize (same behavior as sentence-transformers)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return int(self._dimension)
