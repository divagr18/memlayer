import threading
import json
from functools import lru_cache
import time
from typing import List, Dict, Any, Optional

import dateutil
import numpy as np
from .storage.networkx import NetworkXStorage
from .storage.chroma import ChromaStorage
from .storage.memgraph import MemgraphStorage
from .embedding_models import BaseEmbeddingModel
from .wrappers.base import BaseLLMWrapper
from .observability import Trace
from .ml_gate import SalienceGate
from .storage.base import BaseGraphStorage
from .config import is_debug_mode
class SearchService:
    """
    Handles memory retrieval with built-in caching and observability.
    It is completely agnostic to the specific embedding model used.
    Supports LIGHTWEIGHT mode (graph-only, no vectors).
    """
    def __init__(self, vector_storage: Optional[ChromaStorage], graph_storage: BaseGraphStorage, embedding_model: Optional[BaseEmbeddingModel]):
        self.storage = vector_storage
        self.embedding_model = embedding_model
        self.graph_storage = graph_storage
        self.is_lightweight = (vector_storage is None or embedding_model is None)
        
        # Apply an LRU cache to the embedding generation method (only in non-lightweight mode).
        if not self.is_lightweight:
            self._get_embedding_cached = lru_cache(maxsize=256)(self._get_embedding_uncached)

    def _get_embedding_uncached(self, text: str) -> List[float]:
        return self.embedding_model.get_embeddings([text])[0]
    def _track_attention(self, vector_results: List[Dict], graph_results: List[str]):
        """A helper to call the storage layers to track memory access."""
        if not self.is_lightweight and vector_results:
            vector_ids = [res['id'] for res in vector_results]
            self.storage.track_memory_access(vector_ids)
        
        if graph_results:
            # We need to parse node names from the graph context strings
            node_names = set()
            import re
            for line in graph_results:
                # Regex to find text within parentheses like (Person) John
                matches = re.findall(r'\)\s(.*?)\s--', line)
                node_names.update(matches)
            
            if node_names:
                self.graph_storage.track_memory_access(list(node_names))
    def search(self, query: str, user_id: str, search_tier: str = "balanced", llm_client: Optional[BaseLLMWrapper] = None) -> Dict[str, Any]:
        """
        Performs a hybrid search, combining vector search with graph traversal for "deep" queries.
        In LIGHTWEIGHT mode (no embeddings), only performs graph-based search.
        
        Args:
            query: The search query
            user_id: User identifier
            search_tier: "fast", "balanced", or "deep"
            llm_client: Optional LLM client for entity extraction (required for "deep" search)
        
        Returns:
            A dictionary containing the formatted result string and the trace object.
        """
        trace = Trace()
        vector_results = []
        graph_facts = set()
        try:
            vector_context = ""
            
            # --- STEP 1: VECTOR SEARCH (skip in LIGHTWEIGHT mode) ---
            if not self.is_lightweight:
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
            else:
                # LIGHTWEIGHT mode: skip vector search
                trace.add_event("vector_search", 0, metadata={"skipped": "lightweight_mode"})

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
            with trace.start_event("attention_tracking"):
                self._track_attention(vector_results, list(graph_facts))
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
        Checks for pending tasks that are now due, formats them into a context string,
        and updates their status to 'completed' to prevent re-showing.

        Args:
            user_id (str): The user to check for tasks.

        Returns:
            A formatted string containing reminders for the LLM, or an empty string if none.
        """
        due_tasks = self.graph_storage.get_due_tasks_for_user(user_id)
        
        if not due_tasks:
            return ""

        print(f"[SearchService] Found {len(due_tasks)} due task(s) for user '{user_id}'.")
        
        # Format the tasks into a STRONG, DIRECT system prompt
        # The instruction must be imperative to ensure the LLM includes it
        context = "🚨 CRITICAL SYSTEM INSTRUCTION 🚨\n"
        context += "You MUST inform the user about the following overdue reminders in your NEXT response.\n"
        context += "This is mandatory - do not skip or forget these reminders:\n\n"
        
        for task in due_tasks:
            task_id = task['id']
            description = task.get('description', 'No description.')
            due_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.get('due_timestamp')))
            
            context += f"⏰ REMINDER: {description}\n"
            
            # Mark as completed to prevent re-showing
            self.graph_storage.update_task_status(task_id, 'completed')

        return context.strip()

class ConsolidationService:
    """
    Handles background consolidation of memories. It is agnostic to the specific
    embedding model and LLM provider used for fact extraction.
    Supports LIGHTWEIGHT mode (graph-only, no vectors).
    """
    def __init__(self, vector_storage: Optional[ChromaStorage], graph_storage: MemgraphStorage, embedding_model: Optional[BaseEmbeddingModel], salience_gate: SalienceGate, llm_client: BaseLLMWrapper):
        self.storage = vector_storage
        self.graph_storage = graph_storage
        self.embedding_model = embedding_model
        self.salience_gate = salience_gate
        self.llm_client = llm_client
        self.is_lightweight = (vector_storage is None or embedding_model is None)
        self._consolidation_complete = threading.Event()
        self._consolidation_complete.set()  # Initially set (no consolidation in progress)

    def consolidate(self, conversation_text: str, user_id: str):
        """
        Extracts knowledge using the LLM wrapper and saves facts to vector store,
        entities and relationships to graph store. Runs in a background thread.
        """
        # Mark consolidation as in progress
        self._consolidation_complete.clear()
        
        def _task():
            if is_debug_mode():
                print(f"[DEBUG] Background consolidation thread started for user '{user_id}'")
            print(f"Consolidating knowledge for user '{user_id}'...")
            
            # Check if the conversation is worth saving (salience gate)
            if is_debug_mode():
                print(f"[DEBUG] Checking salience...")
            if not self.salience_gate.is_worth_saving(conversation_text):
                print("Conversation deemed not salient. Skipping consolidation.")
                return
            
            if is_debug_mode():
                print(f"[DEBUG] Salience check passed! Proceeding with extraction...")
            try:
                # 1. Delegate knowledge extraction to the LLM wrapper
                print("Extracting knowledge from conversation...")
                knowledge_graph = self.llm_client.analyze_and_extract_knowledge(conversation_text)
                
                facts = knowledge_graph.get("facts", [])
                entities = knowledge_graph.get("entities", [])
                relationships = knowledge_graph.get("relationships", [])

                print(f"Extracted: {len(facts)} facts, {len(entities)} entities, {len(relationships)} relationships")

                # 2. Process and save facts to the vector store (skip in LIGHTWEIGHT mode)
                if facts and not self.is_lightweight:
                    fact_texts = []
                    embeddings = []
                    metadatas = []
                    
                    # Prepare data for batch processing
                    for fact_info in facts:
                        fact_text = fact_info.get("fact")
                        if not fact_text: continue
                        
                        fact_texts.append(fact_text)
                        
                        expiration_timestamp = None
                        if fact_info.get("expiration_date"):
                            try:
                                expiration_timestamp = dateutil.parser.parse(fact_info["expiration_date"]).timestamp()
                            except:
                                pass # Ignore invalid date formats
                        
                        metadatas.append({
                            "importance_score": fact_info.get("importance_score", 0.5),
                            "expiration_timestamp": expiration_timestamp
                        })

                    if fact_texts:
                        embeddings = self.embedding_model.get_embeddings(fact_texts)
                        for i, fact_text in enumerate(fact_texts):
                            self.storage.add_memory(
                                content=fact_text, 
                                embedding=embeddings[i], 
                                user_id=user_id,
                                metadata=metadatas[i] # <-- Pass the new metadata
                            )
                        print(f"✓ Saved {len(fact_texts)} facts to vector store.")
                elif facts and self.is_lightweight:
                    print(f"* Skipped saving {len(facts)} facts to vector store (LIGHTWEIGHT mode).")

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
            finally:
                # Always mark consolidation as complete (even if error occurred)
                self._consolidation_complete.set()
        
        # Start the consolidation task in a background thread
        thread = threading.Thread(target=_task, daemon=True)
        thread.start()
        if is_debug_mode():
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
class CurationService:
    """A background service that handles memory decay and expiration."""
    def __init__(self, vector_storage: Optional[ChromaStorage], graph_storage: BaseGraphStorage, interval_seconds: int = 3600):
        self.vector_storage = vector_storage
        self.graph_storage = graph_storage
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _calculate_relevance(self, memory: Dict) -> float:
        """Calculates the current relevance score of a memory."""
        now = time.time()
        
        # Age in days (with a minimum of 0.1 to avoid division by zero and handle brand-new memories)
        age_days = max(0.1, (now - memory.get('created_timestamp', now)) / 86400)
        
        # Recency boost (fades over 7 days) - but starts at 0 for brand new memories
        # Only give recency boost if memory has been accessed at least once
        time_since_access = now - memory.get('last_accessed_timestamp', now)
        if memory.get('access_count', 0) > 0:
            recency_boost = max(0, 1 - (time_since_access / (86400 * 7)))
        else:
            recency_boost = 0  # No boost for never-accessed memories
        
        # Attention score (log scaled)
        attention_score = np.log1p(memory.get('access_count', 0))
        
        # Final score - use age_days directly (not log) to give proper weight to age
        # For brand-new memories (age < 1 day), use 1 day as denominator
        age_factor = max(1.0, age_days)
        score = (memory.get('importance_score', 0.5) + attention_score + recency_boost) / age_factor
        return score

    def _run(self):
        print("[CurationService] Background thread started.")
        while not self._stop_event.is_set():
            try:
                print(f"[CurationService] Running memory curation cycle (interval: {self.interval}s)...")
                now = time.time()
                
                # Get all memories from both stores
                all_memories = self.graph_storage.get_all_memories_for_curation()
                if self.vector_storage:
                    all_memories.extend(self.vector_storage.get_all_memories_for_curation())

                print(f"[CurationService] Found {len(all_memories)} total memories to check")
                archived_count = 0
                deleted_count = 0

                for mem in all_memories:
                    mem_id = mem.get('id')
                    # 1. Librarian: Check for hard expiration
                    exp_ts = mem.get('expiration_timestamp')
                    print(f"[CurationService] Memory '{mem_id}': exp_ts={exp_ts}, now={now}")
                    if exp_ts and exp_ts < now:
                        print(f"[CurationService] DELETING '{mem_id}' (expired at {exp_ts}, now is {now})")
                        self.graph_storage.delete_memory(mem_id)
                        if self.vector_storage: self.vector_storage.delete_memory(mem_id)
                        deleted_count += 1
                        continue

                    # 2. Gardener: Check for decay if status is active
                    if mem.get('status') == 'active':
                        relevance = self._calculate_relevance(mem)
                        print(f"[CurationService] Checking '{mem_id}': relevance={relevance:.3f}, importance={mem.get('importance_score')}, access_count={mem.get('access_count')}")
                        if relevance < 0.3: # Configurable threshold
                            print(f"[CurationService] ARCHIVING '{mem_id}' (relevance {relevance:.3f} < 0.3)")
                            self.graph_storage.update_memory_status(mem_id, 'archived')
                            if self.vector_storage: self.vector_storage.update_memory_status(mem_id, 'archived')
                            archived_count += 1
                
                print(f"[CurationService] Cycle complete. Archived: {archived_count}, Deleted: {deleted_count}.")

            except Exception as e:
                print(f"[CurationService] Error in background thread: {e}")
                import traceback
                traceback.print_exc()
            
            self._stop_event.wait(self.interval)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)