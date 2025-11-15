import time
import uuid
import networkx as nx
from pathlib import Path
import pickle
import threading
from typing import List, Dict, Any
from .base import BaseGraphStorage

class NetworkXStorage(BaseGraphStorage):
    """
    A truly embedded graph storage backend using NetworkX.
    The graph is held in memory for fast queries and is automatically persisted
    to a file on disk after every modification to ensure data durability.
    """
    def __init__(self, storage_path: str):
        self.graph_path = Path(storage_path) / "knowledge_graph.pkl"
        self._lock = threading.Lock()  # To prevent race conditions during save operations
        self.graph: nx.DiGraph = self._load_graph()
        
        print(
            f"Memory Bank (NetworkX) initialized. "
            f"Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges "
            f"from: {self.graph_path}"
        )

    def _load_graph(self) -> nx.DiGraph:
        """
        Loads the graph from a pickle file. If the file doesn't exist or is
        corrupt, it creates a new, empty graph.
        """
        if self.graph_path.exists():
            try:
                with self.graph_path.open('rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                print(
                    f"Warning: Could not load graph file at {self.graph_path}. "
                    f"It might be corrupted. Starting with a fresh graph. Error: {e}"
                )
                # Optionally, create a backup of the corrupted file
                backup_path = self.graph_path.with_suffix('.pkl.corrupted')
                self.graph_path.rename(backup_path)
                print(f"Backed up corrupted graph to {backup_path}")
        
        return nx.DiGraph()

    def _save_graph(self):
        """
        Atomically saves the current in-memory graph to a pickle file.
        It writes to a temporary file first and then renames it to avoid
        corruption if the process is interrupted mid-write.
        """
        with self._lock:
            temp_path = self.graph_path.with_suffix('.pkl.tmp')
            try:
                self.graph_path.parent.mkdir(parents=True, exist_ok=True)
                with temp_path.open('wb') as f:
                    pickle.dump(self.graph, f)
                
                # On Windows, we need to remove the target file before renaming
                # because os.rename() doesn't overwrite on Windows
                if self.graph_path.exists():
                    self.graph_path.unlink()
                
                temp_path.rename(self.graph_path)
            except Exception as e:
                print(f"FATAL: Could not save knowledge graph to {self.graph_path}. Error: {e}")
    
    def _find_canonical_entity(self, name: str, node_type: str = "Concept", similarity_threshold: float = 0.85) -> str:
        """
        Finds the canonical (existing) entity name that matches the given name.
        This prevents duplicate entities like "Dr. Watson" and "Dr. Emma Watson".
        
        Strategy:
        1. Exact match (case-insensitive) -> return existing
        2. Substring match (one is substring of other) -> prefer longer name
        3. High similarity match -> prefer existing node
        4. No match -> return original name (will create new node)
        
        Args:
            name: The entity name to check
            node_type: The type of entity (Person, Organization, etc.)
            similarity_threshold: Minimum similarity to consider a match (0.0-1.0)
        
        Returns:
            The canonical entity name to use (either existing or new)
        """
        if not name or not name.strip():
            return name
        
        name_lower = name.lower().strip()
        
        # Get all existing nodes of the same type
        same_type_nodes = [n for n, data in self.graph.nodes(data=True) 
                          if data.get('type') == node_type]
        
        if not same_type_nodes:
            return name  # No existing entities of this type
        
        # Strategy 1: Exact match (case-insensitive)
        for existing_node in same_type_nodes:
            if existing_node.lower() == name_lower:
                return existing_node  # Use existing name (preserves original casing)
        
        # Strategy 2: Word-based matching with preference for full names
        # "Dr. Watson" vs "Dr. Emma Watson" -> use "Dr. Emma Watson"
        # Split into words for comparison
        name_words = name_lower.split()
        name_words_set = set(name_words)
        
        for existing_node in same_type_nodes:
            existing_words = existing_node.lower().split()
            existing_words_set = set(existing_words)
            
            # Check if all words from one name are in the other (subset relationship)
            if existing_words_set.issubset(name_words_set):
                # Existing name is shorter (e.g., "Dr. Watson" when adding "Dr. Emma Watson")
                # New name is longer - merge existing into new
                if len(name_words) > len(existing_words):
                    self._merge_entity_nodes(existing_node, name)
                    return name
                else:
                    # Same length but reordered - use existing
                    return existing_node
            
            elif name_words_set.issubset(existing_words_set):
                # New name is shorter (e.g., "Dr. Watson" when "Dr. Emma Watson" exists)
                # Use the existing longer name
                if len(existing_words) > len(name_words):
                    return existing_node
                else:
                    # Same length but reordered - use existing
                    return existing_node
        
        # Strategy 3: High similarity match (word overlap for non-subset cases)
        # "Emma Watson" vs "E. Watson" -> same person
        best_match = None
        best_similarity = 0.0
        
        for existing_node in same_type_nodes:
            existing_words = set(existing_node.lower().split())
            
            # Jaccard similarity
            intersection = len(name_words_set & existing_words)
            union = len(name_words_set | existing_words)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_node
        
        if best_match:
            # High similarity found - use the longer name as canonical
            if len(name) > len(best_match):
                self._merge_entity_nodes(best_match, name)
                return name
            else:
                return best_match
        
        # No match found - this is a genuinely new entity
        return name
    
    def _merge_entity_nodes(self, old_name: str, new_name: str):
        """
        Merges two entity nodes by renaming the old node to the new name
        and consolidating all edges.
        
        Args:
            old_name: The existing node name to merge from
            new_name: The canonical name to merge into
        """
        if not self.graph.has_node(old_name):
            return
        
        if old_name == new_name:
            return
        
        # Get node attributes from old node
        old_attrs = self.graph.nodes[old_name].copy()
        
        # Create new node if it doesn't exist, or update its type
        if not self.graph.has_node(new_name):
            self.graph.add_node(new_name, **old_attrs)
        
        # Move all edges from old node to new node
        # Incoming edges
        for predecessor in list(self.graph.predecessors(old_name)):
            edge_data = self.graph.get_edge_data(predecessor, old_name)
            if not self.graph.has_edge(predecessor, new_name):
                self.graph.add_edge(predecessor, new_name, **edge_data)
        
        # Outgoing edges
        for successor in list(self.graph.successors(old_name)):
            edge_data = self.graph.get_edge_data(old_name, successor)
            if not self.graph.has_edge(new_name, successor):
                self.graph.add_edge(new_name, successor, **edge_data)
        
        # Remove old node (this also removes its edges)
        self.graph.remove_node(old_name)
        
        print(f"[Entity Deduplication] Merged '{old_name}' into '{new_name}'")

    def add_entity(self, name: str, node_type: str = "Concept") -> str:
        """

        Adds a node to the graph if it doesn't already exist.
        Automatically detects and merges duplicate entities (e.g., "Dr. Watson" and "Dr. Emma Watson").
        Node attributes are stored in a dictionary associated with the node.
        This operation is immediately persisted to disk.
        
        Returns:
            The canonical entity name that was used (may differ from input due to deduplication)
        """
        if not name or not name.strip():
            return name
        
        # Find canonical entity name (handles deduplication)
        canonical_name = self._find_canonical_entity(name, node_type)
        
        # NetworkX nodes are unique. Adding an existing node updates its attributes.
        if self.graph.has_node(canonical_name):
            # If you want to update the type if it changes, you can add this:
            # self.graph.nodes[canonical_name]['type'] = node_type
            pass # Node already exists, do nothing to prevent overwriting type
        else:
            self.graph.add_node(canonical_name, type=node_type)
            self._save_graph()
        
        return canonical_name

    def add_relationship(self, subject_name: str, predicate: str, object_name: str):
        """
        Adds a directed edge between two nodes, creating the nodes if they
        don't exist. The relationship type is stored as an edge attribute.
        Automatically handles entity deduplication.
        This operation is immediately persisted to disk.
        """
        if not all([subject_name, predicate, object_name]):
            return
        
        # Ensure nodes exist and get their canonical names (handles deduplication)
        canonical_subject = self.add_entity(subject_name)
        canonical_object = self.add_entity(object_name)
        
        # Add the edge with the predicate as an attribute using canonical names
        if not self.graph.has_edge(canonical_subject, canonical_object):
            self.graph.add_edge(canonical_subject, canonical_object, type=predicate)
            self._save_graph()

    def get_related_concepts(self, concept_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Finds immediate neighbors of a given node in the graph, describing
        the relationship from the perspective of the starting node.
        """
        if not self.graph.has_node(concept_name):
            return []
        
        related = []
        
        # Get successors (outgoing relationships)
        for successor in self.graph.successors(concept_name):
            if len(related) >= limit: break
            edge_data = self.graph.get_edge_data(concept_name, successor)
            node_data = self.graph.nodes[successor]
            related.append({
                "name": successor,
                "relationship_type": edge_data.get('type', 'related to'),
                "type": node_data.get('type', 'Concept')
            })

        # Get predecessors (incoming relationships)
        for predecessor in self.graph.predecessors(concept_name):
            if len(related) >= limit: break
            edge_data = self.graph.get_edge_data(predecessor, concept_name)
            node_data = self.graph.nodes[predecessor]
            related.append({
                "name": predecessor,
                "relationship_type": f"is {edge_data.get('type', 'related to')} of", # Invert relationship
                "type": node_data.get('type', 'Concept')
            })
            
        return related[:limit]
    
    def find_matching_nodes(self, query_name: str, threshold: float = 0.7) -> List[str]:
        """
        Finds nodes in the graph that match the query name.
        Uses fuzzy matching to handle variations like "Dr. Emma Watson" vs "Emma Watson".
        
        Args:
            query_name (str): The entity name to search for
            threshold (float): Similarity threshold (0.0-1.0). Default 0.7.
        
        Returns:
            List of matching node names, sorted by similarity (best match first)
        """
        if not query_name or not query_name.strip():
            return []
        
        # First try exact match (case-insensitive)
        query_lower = query_name.lower()
        for node in self.graph.nodes():
            if node.lower() == query_lower:
                return [node]  # Exact match, return immediately
        
        # Try partial matching (substring)
        matches = []
        query_words = set(query_lower.split())
        
        for node in self.graph.nodes():
            node_lower = node.lower()
            node_words = set(node_lower.split())
            
            # Check if query is substring of node or vice versa
            if query_lower in node_lower or node_lower in query_lower:
                matches.append((node, 1.0))  # High score for substring match
                continue
            
            # Calculate word overlap similarity (Jaccard similarity)
            if query_words and node_words:
                intersection = len(query_words & node_words)
                union = len(query_words | node_words)
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity >= threshold:
                    matches.append((node, similarity))
        
        # Sort by similarity (descending) and return just the node names
        matches.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in matches]
    
    def get_subgraph_context(self, concept_name: str, depth: int = 1) -> List[str]:
        """
        Retrieves a textual representation of the subgraph surrounding a concept.
        Traverses 'depth' hops out from the start node and formats the relationships
        into a list of strings for an LLM to easily understand.

        Args:
            concept_name (str): The name of the entity to start the traversal from.
            depth (int): The number of hops to explore (radius). Default is 1.

        Returns:
            A list of strings, where each string describes a relationship.
            Example: ["(Person) John --[is lead engineer for]--> (Project) Project Phoenix"]
        """
        if not self.graph.has_node(concept_name):
            return []

        # Use ego_graph to efficiently get the subgraph of neighbors within the given radius.
        # This includes the start node itself.
        subgraph = nx.ego_graph(self.graph, concept_name, radius=depth)
        
        relationships = []
        # Iterate over all edges in the extracted subgraph
        for u, v, data in subgraph.edges(data=True):
            rel_type = data.get('type', 'related to')
            
            # Get node types for richer context
            u_type = self.graph.nodes[u].get('type', 'Concept')
            v_type = self.graph.nodes[v].get('type', 'Concept')
            
            # Format into a clear, readable string
            relationships.append(f"({u_type}) {u} --[{rel_type}]--> ({v_type}) {v}")
            
        return relationships
    def add_task(self, description: str, due_timestamp: float, user_id: str) -> str:
        """
        Adds a new 'Task' node to the knowledge graph.

        Args:
            description (str): The text description of the task.
            due_timestamp (float): The Unix timestamp when the task is due.
            user_id (str): The user this task belongs to.

        Returns:
            The unique ID of the newly created task node.
        """
        if not description or not due_timestamp:
            return ""
        
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        # Add the task as a new node with specific attributes
        self.graph.add_node(
            task_id,
            type='Task',
            description=description,
            due_timestamp=due_timestamp,
            status='pending',
            user_id=user_id,
            created_timestamp=time.time()
        )
        
        # A task is a type of memory, so it should be connected to the user
        # This allows for easy retrieval of all tasks for a user.
        self.add_entity(user_id, node_type="User")
        self.add_relationship(user_id, "has_task", task_id)
        
        self._save_graph()
        print(f"✓ Saved new task '{task_id}' to graph store.")
        return task_id

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Retrieves all tasks from the graph that are currently in 'pending' status.
        """
        pending_tasks = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'Task' and data.get('status') == 'pending':
                task_data = data.copy()
                task_data['id'] = node # Include the node's ID in the result
                pending_tasks.append(task_data)
        return pending_tasks

    def update_task_status(self, task_id: str, new_status: str) -> bool:
        """
        Updates the status of a specific task node.

        Args:
            task_id (str): The unique ID of the task node to update.
            new_status (str): The new status (e.g., 'triggered', 'completed').

        Returns:
            True if the update was successful, False otherwise.
        """
        if self.graph.has_node(task_id) and self.graph.nodes[task_id].get('type') == 'Task':
            self.graph.nodes[task_id]['status'] = new_status
            self.graph.nodes[task_id]['updated_timestamp'] = time.time()
            self._save_graph()
            return True
        return False
    def get_triggered_tasks_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all tasks for a specific user that are in 'triggered' status.
        This is a key method for proactive reminders.

        Args:
            user_id (str): The user to check for triggered tasks.

        Returns:
            A list of task dictionaries, each including the task's ID.
        """
        triggered_tasks = []
        # We can't query by attribute directly in NetworkX efficiently, so we iterate.
        # For performance on huge graphs, this could be optimized with indices,
        # but for thousands of tasks, this is perfectly acceptable.
        for node, data in self.graph.nodes(data=True):
            if (data.get('type') == 'Task' and
                data.get('status') == 'triggered' and
                data.get('user_id') == user_id):
                
                task_data = data.copy()
                task_data['id'] = node
                triggered_tasks.append(task_data)
        
        return triggered_tasks