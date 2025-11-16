from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLMWrapper(ABC):
    """
    Abstract base class defining the common interface for all LLM wrappers.
    This ensures that core services like ConsolidationService can interact
    with any supported LLM in a consistent way.
    """
    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        """
        The primary method for interacting with the LLM, including the full
        tool-use loop for memory.
        """
        pass

    @abstractmethod
    def analyze_and_extract_knowledge(self, text: str) -> Dict:
        """
        Analyzes text to extract a structured knowledge graph...
        ...and now also includes 'importance_score' and 'expiration_date' for facts.
        """
        pass
    @abstractmethod
    def extract_query_entities(self, query: str) -> List[str]:
        """
        Extracts key entities from a short search query to use for graph lookups.
        This should be a fast, focused operation.
        """
        pass