import threading
import json
from functools import lru_cache
import time
from typing import List, Dict, Any, Optional
from .storage.networkx import NetworkXStorage
from .storage.chroma import ChromaStorage
from .storage.memgraph import MemgraphStorage
from .embedding_models import BaseEmbeddingModel
from .wrappers.base import BaseLLMWrapper
from .observability import Trace
from .ml_gate import SalienceGate
from .storage.base import BaseGraphStorage
class SearchService:
    """
    Handles memory retrieval with built-in caching and observability.
    It is completely agnostic to the specific embedding model used.
    """
    def __init__(self, vector_storage: ChromaStorage, graph_storage: BaseGraphStorage,embedding_model: BaseEmbeddingModel):
        self.storage = vector_storage
        self.embedding_model = embedding_model
        self.graph_storage = graph_storage
        # Apply an LRU cache to the embedding generation method.
        # This cache is tied to the instance of the SearchService.
        self._get_embedding_cached = lru_cache(maxsize=256)(self._get_embedding_uncached)

    def _get_embedding_uncached(self, text: str) -> List[float]:
        return self.embedding_model.get_embeddings([text])[0]

    def search(self, query: str, user_id: str, search_tier: str = "balanced", llm_client: Optional[BaseLLMWrapper] = None) -> Dict[str, Any]:
        """
        Performs a hybrid search, combining vector search with graph traversal for "deep" queries.
        
        Args:
            query: The search query
            user_id: User identifier
            search_tier: "fast", "balanced", or "deep"
            llm_client: Optional LLM client for entity extraction (required for "deep" search)
        
        Returns:
            A dictionary containing the formatted result string and the trace object.
        """
        trace = Trace()
        
        try:
            # --- STEP 1: ALWAYS PERFORM VECTOR SEARCH ---
            # --- TRACING EMBEDDING GENERATION (WITH CACHE CHECK) ---
            start_time = time.perf_counter()
            cache_info_before = self._get_embedding_cached.cache_info()
            
            query_embedding = self._get_embedding_cached(query)
            
            cache_info_after = self._get_embedding_cached.cache_info()
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            cache_hit = "hit" if cache_info_after.hits > cache_info_before.hits else "miss"
            trace.add_event(
                "embedding_generation",
                duration_ms,
                metadata={"cache_status": cache_hit}
            )
            # -----------------------------------------------------------

            # --- TRACING VECTOR SEARCH ---
            top_k = 5
            if search_tier == "fast": top_k = 2
            elif search_tier == "deep": top_k = 10
            
            with trace.start_event("vector_search", metadata={"tier": search_tier, "top_k": top_k}) as event:
                results = self.storage.search_memories(
                    query_embedding=query_embedding,
                    user_id=user_id,
                    top_k=top_k
                )
                event.metadata["results_found"] = len(results)
            # -----------------------------------------------------------

            # --- TRACING RESULT FORMATTING ---
            with trace.start_event("result_formatting"):
                if not results:
                    vector_context = "No relevant memories found in vector search."
                else:
                    vector_context = "Relevant memories from vector search:\n"
                    for res in results:
                        vector_context += f"- {res['content']} (Similarity: {res['score']:.2f})\n"
            # -----------------------------------------------------------

            # --- STEP 2: PERFORM GRAPH SEARCH FOR "DEEP" TIER ---
            graph_context = ""
            if search_tier == "deep":
                if not llm_client:
                    print("Warning: 'deep' search tier requested but no llm_client provided for entity extraction. Skipping graph search.")
                else:
                    with trace.start_event("graph_search") as graph_event:
                        # A. Extract entities from the user's query
                        query_entities = llm_client.extract_query_entities(query)
                        graph_event.metadata["extracted_entities"] = query_entities
                        
                        # B. Traverse the graph for each found entity
                        graph_facts = set()  # Use a set to avoid duplicate relationships
                        matched_entities = []  # Track which entities were found in graph
                        all_traversed_nodes = set()  # Track all nodes we've explored
                        
                        # Strategy 1: Start from query entities with 2-hop traversal
                        for entity in query_entities:
                            # Use fuzzy matching to find nodes that match the entity
                            matching_nodes = self.graph_storage.find_matching_nodes(entity, threshold=0.6)
                            
                            if matching_nodes:
                                matched_node = matching_nodes[0]
                                matched_entities.append(f"{entity} -> {matched_node}")
                                all_traversed_nodes.add(matched_node)
                                
                                # Use depth=2 for richer context (2-hop neighbors)
                                facts = self.graph_storage.get_subgraph_context(matched_node, depth=2)
                                graph_facts.update(facts)
                                
                                # WORKAROUND: Also check if there's a "User" node (common extraction artifact)
                                # The "User" node often contains relationships that should belong to the actual person
                                if self.graph_storage.graph.has_node("User") and "User" not in all_traversed_nodes:
                                    user_facts = self.graph_storage.get_subgraph_context("User", depth=1)
                                    if user_facts:
                                        graph_facts.update(user_facts)
                                        all_traversed_nodes.add("User")
                                        matched_entities.append(f"User (related to {matched_node})")
                            else:
                                # Fallback to exact match (original behavior)
                                facts = self.graph_storage.get_subgraph_context(entity, depth=2)
                                if facts:
                                    matched_entities.append(f"{entity} (exact)")
                                    all_traversed_nodes.add(entity)
                                    graph_facts.update(facts)
                        
                        # Strategy 2: Extract entities from vector search results and traverse from them too
                        # This finds related concepts even if not directly connected to query entities
                        if results:
                            additional_entities = set()
                            for result in results[:3]:  # Check top 3 most relevant memories
                                content = result.get('content', '')
                                # Simple entity extraction: Look for capitalized phrases
                                import re
                                # Find sequences of capitalized words (likely proper nouns)
                                potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
                                
                                # Filter out common titles, articles, and short words
                                stopwords = {'Dr', 'Mr', 'Ms', 'Mrs', 'Prof', 'The', 'A', 'An', 
                                           'In', 'On', 'At', 'To', 'For', 'With', 'By', 'From', 'Of',
                                           'My', 'Your', 'His', 'Her', 'Their', 'Our', 'Its', 'This', 'That',
                                           'I', 'We', 'You', 'He', 'She', 'They', 'It'}
                                
                                for entity in potential_entities:
                                    # Skip if it's a stopword or too short
                                    if entity not in stopwords and len(entity) > 2:
                                        additional_entities.add(entity)
                            
                            # Remove entities we've already traversed
                            additional_entities = {e for e in additional_entities if e not in all_traversed_nodes}
                            
                            # Limit to avoid explosion
                            for entity in list(additional_entities)[:5]:
                                matching_nodes = self.graph_storage.find_matching_nodes(entity, threshold=0.7)
                                if matching_nodes:
                                    matched_node = matching_nodes[0]
                                    if matched_node not in all_traversed_nodes:
                                        all_traversed_nodes.add(matched_node)
                                        facts = self.graph_storage.get_subgraph_context(matched_node, depth=1)
                                        if facts:
                                            matched_entities.append(f"{entity} -> {matched_node} (from vector)")
                                            graph_facts.update(facts)
                        
                        if graph_facts:
                            graph_context = "\n\nRelated knowledge from graph:\n" + "\n".join(sorted(list(graph_facts)))
                            graph_event.metadata["relationships_found"] = len(graph_facts)
                            graph_event.metadata["matched_entities"] = matched_entities
                            graph_event.metadata["nodes_traversed"] = len(all_traversed_nodes)
                        else:
                            graph_event.metadata["relationships_found"] = 0
                            graph_event.metadata["matched_entities"] = []
                            graph_event.metadata["nodes_traversed"] = 0
            # -----------------------------------------------------------

            # --- STEP 3: COMBINE CONTEXTS AND CONCLUDE ---
            final_result = f"{vector_context}{graph_context}".strip()
            trace.conclude(result=final_result)

        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            trace.conclude(error=e)
            final_result = "An error occurred while searching memory."

        return {"result": final_result, "trace": trace}
    def get_triggered_tasks_context(self, user_id: str) -> str:
        """
        Checks for triggered tasks for a user, formats them into a context string,
        and updates their status to 'completed' to prevent re-triggering.

        Args:
            user_id (str): The user to check for tasks.

        Returns:
            A formatted string containing reminders for the LLM, or an empty string if none.
        """
        triggered_tasks = self.graph_storage.get_triggered_tasks_for_user(user_id)
        
        if not triggered_tasks:
            return ""

        print(f"[SearchService] Found {len(triggered_tasks)} triggered tasks for user '{user_id}'.")
        
        # Format the tasks into a system prompt
        context = "System Note: The following scheduled tasks are now due and require the user's attention. Please inform the user about them naturally in your response.\n"
        
        for task in triggered_tasks:
            task_id = task['id']
            description = task.get('description', 'No description.')
            due_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.get('due_timestamp')))
            
            context += f"- Task: '{description}' (Originally scheduled for {due_date})\n"
            
            # --- CRITICAL: Update status to prevent re-triggering ---
            # We mark it as 'completed' assuming the LLM's notification fulfills the task.
            # A more advanced system might use 'acknowledged' and require user confirmation.
            self.graph_storage.update_task_status(task_id, 'completed')

        return context.strip()

class ConsolidationService:
    """
    Handles background consolidation of memories. It is agnostic to the specific
    embedding model and LLM provider used for fact extraction.
    """
    def __init__(self, vector_storage: ChromaStorage, graph_storage: MemgraphStorage, embedding_model: BaseEmbeddingModel, salience_gate: SalienceGate, llm_client: BaseLLMWrapper):
        self.storage = vector_storage
        self.graph_storage = graph_storage
        self.embedding_model = embedding_model
        self.salience_gate = salience_gate
        self.llm_client = llm_client

    def consolidate(self, conversation_text: str, user_id: str):
        """
        Extracts knowledge using the LLM wrapper and saves facts to vector store,
        entities and relationships to graph store. Runs in a background thread.
        """
        def _task():
            print(f"[DEBUG] Background consolidation thread started for user '{user_id}'")
            print(f"Consolidating knowledge for user '{user_id}'...")
            
            # Check if the conversation is worth saving (salience gate)
            print(f"[DEBUG] Checking salience...")
            if not self.salience_gate.is_worth_saving(conversation_text):
                print("Conversation deemed not salient. Skipping consolidation.")
                return
            
            print(f"[DEBUG] Salience check passed! Proceeding with extraction...")
            try:
                # 1. Delegate knowledge extraction to the LLM wrapper
                print("Extracting knowledge from conversation...")
                knowledge_graph = self.llm_client.analyze_and_extract_knowledge(conversation_text)
                
                facts = knowledge_graph.get("facts", [])
                entities = knowledge_graph.get("entities", [])
                relationships = knowledge_graph.get("relationships", [])

                print(f"Extracted: {len(facts)} facts, {len(entities)} entities, {len(relationships)} relationships")

                # 2. Process and save facts to the vector store
                if facts:
                    fact_texts = [f["fact"] for f in facts if f.get("fact")]
                    if fact_texts:
                        embeddings = self.embedding_model.get_embeddings(fact_texts)
                        for i, fact_text in enumerate(fact_texts):
                            self.storage.add_memory(
                                content=fact_text, embedding=embeddings[i], user_id=user_id
                            )
                        print(f"✓ Saved {len(fact_texts)} facts to vector store.")

                # 3. Process and save entities and relationships to the graph store
                if entities:
                    for entity in entities:
                        self.graph_storage.add_entity(name=entity.get("name"), node_type=entity.get("type", "Concept"))
                    print(f"✓ Saved {len(entities)} entities to graph store.")
                
                if relationships:
                    for rel in relationships:
                        self.graph_storage.add_relationship(
                            subject_name=rel.get("subject"),
                            predicate=rel.get("predicate"),
                            object_name=rel.get("object")
                        )
                    print(f"✓ Saved {len(relationships)} relationships to graph store.")
                
                print(f"Consolidation complete for user '{user_id}'.")

            except Exception as e:
                print(f"An unexpected error during the consolidation task: {e}")
                import traceback
                traceback.print_exc()
        
        # Start the consolidation task in a background thread
        thread = threading.Thread(target=_task, daemon=True)
        thread.start()
        print(f"[DEBUG] Started background consolidation thread for user '{user_id}'")

class SchedulerService:
    """
    A background service that periodically checks for and triggers scheduled tasks.
    """
    def __init__(self, graph_storage: NetworkXStorage, check_interval_seconds: int = 60):
        """
        Initializes the SchedulerService.

        Args:
            graph_storage: The graph storage instance to query for tasks.
            check_interval_seconds (int): How often to check for due tasks.
        """
        self.graph_storage = graph_storage
        self.check_interval = check_interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        """The main loop for the background thread."""
        print("[SchedulerService] Background thread started.")
        while not self._stop_event.is_set():
            try:
                # --- Check for due tasks ---
                now_timestamp = time.time()
                pending_tasks = self.graph_storage.get_pending_tasks()
                
                if pending_tasks:
                    print(f"[SchedulerService] Checking {len(pending_tasks)} pending tasks...")
                    for task in pending_tasks:
                        if task.get('due_timestamp', float('inf')) <= now_timestamp:
                            print(f"[SchedulerService] Task '{task['id']}' is due. Triggering...")
                            self.graph_storage.update_task_status(task['id'], 'triggered')
                
            except Exception as e:
                print(f"[SchedulerService] Error in background thread: {e}")
            
            # Wait for the next interval, but check for the stop event frequently.
            # This makes shutdown much more responsive.
            self._stop_event.wait(self.check_interval)
        
        print("[SchedulerService] Background thread stopped.")

    def start(self):
        """Starts the background scheduler thread."""
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        """Signals the background scheduler thread to stop."""
        print("[SchedulerService] Stopping background thread...")
        self._stop_event.set()
        self._thread.join(timeout=5)