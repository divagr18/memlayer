from gqlalchemy import Memgraph
from typing import List, Dict, Any
import warnings
import logging

# Suppress connection warnings from GQLAlchemy/Neo4j driver
logging.getLogger('neo4j').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*socket.*')

# --- Graph Schema Definition ---
# We use raw Cypher queries for all operations, so we don't need to define
# OGM classes with GQLAlchemy's Node and Relationship base classes.
# This avoids the database connection requirement at import time.
# The schema is enforced through our Cypher queries using MERGE operations.

# --------------------------------

class MemgraphStorage:
    """
    A graph storage backend using Memgraph via GQLAlchemy.
    This class handles all interactions with the knowledge graph.
    """
    def __init__(self, storage_path: str = None, host: str = "127.0.0.1", port: int = 7687, 
                 username: str = "", password: str = ""):
        """
        Initializes the Memgraph database connection.
        
        Args:
            storage_path: Not used for Memgraph (server-based), kept for API compatibility
            host: Memgraph server host (default: 127.0.0.1)
            port: Memgraph server port (default: 7687)
            username: Memgraph username (default: empty)
            password: Memgraph password (default: empty)
        """
        # Connect to Memgraph server
        self.db = None
        try:
            db = Memgraph(host=host, port=port, username=username, password=password)
            # Test the connection with a simple query
            db.execute("RETURN 1")
            self.db = db
            print(f"MemLayer (Memgraph) connected to {host}:{port}")
            
            # Create indices for better query performance
            try:
                self.db.execute("CREATE INDEX ON :KnowledgeNode(name);")
            except Exception:
                pass  # Index might already exist
            
            try:
                self.db.execute("CREATE INDEX ON :KnowledgeNode(node_type);")
            except Exception:
                pass  # Index might already exist
                
        except Exception as e:
            print(f"Warning: Could not connect to Memgraph at {host}:{port}.")
            print("Knowledge graph features will be disabled. To enable, start Memgraph server.")
            self.db = None

    def add_entity(self, name: str, node_type: str = "Concept"):
        """
        Adds or updates an entity (a node) in the knowledge graph.
        This method is idempotent: if a node with the given name already
        exists, it will not be duplicated.

        Args:
            name (str): The unique name of the entity (e.g., "Project Phoenix").
            node_type (str): The type of the entity (e.g., "Project").
        """
        if not self.db:
            return
        
        if not name or not name.strip():
            return
        
        # We use a raw Cypher query with MERGE for robust idempotency.
        # MERGE finds a pattern or creates it if it doesn't exist.
        # ON CREATE sets properties only if the node is being created for the first time.
        query = "MERGE (n:KnowledgeNode {name: $name}) ON CREATE SET n.node_type = $node_type"
        self.db.execute(query, {"name": name, "node_type": node_type})

    def add_relationship(self, subject_name: str, predicate: str, object_name: str):
        """
        Adds a directed relationship (an edge) between two existing entities.
        This method is idempotent.

        Args:
            subject_name (str): The name of the starting node.
            predicate (str): The type of the relationship (e.g., "is lead engineer for").
            object_name (str): The name of the ending node.
        """
        if not self.db:
            return
            
        if not all([subject_name, predicate, object_name]):
            return

        # This Cypher query finds the two nodes and creates the relationship
        # between them if it doesn't already exist.
        query = """
        MATCH (subject:KnowledgeNode {name: $subject_name})
        MATCH (object:KnowledgeNode {name: $object_name})
        MERGE (subject)-[r:RELATIONSHIP {type: $predicate}]->(object)
        """
        self.db.execute(query, {
            "subject_name": subject_name,
            "predicate": predicate,
            "object_name": object_name
        })

    def get_related_concepts(self, concept_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Finds concepts related to a given concept name by traversing the graph.

        Args:
            concept_name (str): The name of the starting concept.
            limit (int): The maximum number of results to return.

        Returns:
            A list of dictionaries, each representing a related concept and the relationship.
        """
        if not self.db:
            return []
            
        if not concept_name:
            return []
            
        # This query finds the starting node and then traverses any incoming or
        # outgoing relationships to find its neighbors.
        query = """
        MATCH (start_node:KnowledgeNode {name: $concept_name})-[r:RELATIONSHIP]-(related_node:KnowledgeNode)
        RETURN related_node.name AS name, r.type AS relationship_type, related_node.node_type as type
        LIMIT $limit
        """
        results = self.db.execute_and_fetch(query, {"concept_name": concept_name, "limit": limit})
        return list(results)

    def get_all_data_for_test(self) -> Dict[str, List]:
        """
        A utility method for testing and debugging. Fetches all nodes and
        relationships from the graph.
        """
        if not self.db:
            return {"nodes": [], "relationships": []}
            
        nodes_query = "MATCH (n:KnowledgeNode) RETURN n.name AS name, n.node_type AS type"
        nodes = list(self.db.execute_and_fetch(nodes_query))
        
        rels_query = """
        MATCH (subject:KnowledgeNode)-[r:RELATIONSHIP]->(object:KnowledgeNode)
        RETURN subject.name AS subject, r.type AS predicate, object.name AS object
        """
        relationships = list(self.db.execute_and_fetch(rels_query))
        
        return {"nodes": nodes, "relationships": relationships}