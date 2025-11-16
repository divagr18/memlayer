from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseGraphStorage(ABC):
    """
    Abstract base class defining the common interface for all graph storage backends.
    This ensures that services like ConsolidationService can interact with any
    graph implementation in a consistent way.
    """
    @abstractmethod
    def add_entity(self, name: str, node_type: str = "Concept"):
        """Adds a node to the graph, avoiding duplicates."""
        pass

    @abstractmethod
    def add_relationship(self, subject_name: str, predicate: str, object_name: str):
        """Adds a directed edge between two nodes."""
        pass

    @abstractmethod
    def get_related_concepts(self, concept_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves nodes connected to a given node."""
        pass

    @abstractmethod
    def get_subgraph_context(self, concept_name: str, depth: int = 1) -> List[str]:
        """
        Retrieves a textual representation of the subgraph surrounding a concept.
        Traverses 'depth' hops out from the start node and formats the relationships
        into a list of strings for an LLM to easily understand.
        """
        pass