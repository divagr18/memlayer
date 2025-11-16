import anthropic
import json
from typing import Any, List, Dict, Optional, TYPE_CHECKING

# Use TYPE_CHECKING to avoid slow imports at module load time
if TYPE_CHECKING:
    from ..storage.chroma import ChromaStorage
    from ..storage.networkx import NetworkXStorage
    from ..storage.memgraph import MemgraphStorage
    from ..embedding_models import BaseEmbeddingModel, LocalEmbeddingModel
    from ..ml_gate import SalienceGate
    from ..services import SearchService, ConsolidationService, CurationService
    from ..observability import Trace
    from .base import BaseLLMWrapper
else:
    ChromaStorage = None
    NetworkXStorage = None
    MemgraphStorage = None
    BaseEmbeddingModel = None
    LocalEmbeddingModel = None
    SalienceGate = None
    SearchService = None
    ConsolidationService = None
    CurationService = None
    BaseLLMWrapper = object


class Claude(BaseLLMWrapper):
    """
    A memory-enhanced Anthropic Claude client that can be used standalone.
    
    Usage:
        from memlayer.wrappers.claude import Claude
        
        client = Claude(
            api_key="your-api-key",
            model="claude-3-5-sonnet-20241022",
            storage_path="./my_memories",
            user_id="user_123"
        )
        
        response = client.chat(messages=[
            {"role": "user", "content": "What's my favorite color?"}
        ])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        storage_path: str = "./memlayer_data",
        user_id: str = "default_user",
        embedding_model: Optional["BaseEmbeddingModel"] = None,
        salience_threshold: float = 0.0,
        operation_mode: str = "online",  # "local", "online", or "lightweight"
        scheduler_interval_seconds: int = 60,  # For tasks
        curation_interval_seconds: int = 3600,  # For curation
        **kwargs
    ):
        """
        Initialize a memory-enhanced Claude client.
        
        Args:
            api_key: Anthropic API key (if None, will use ANTHROPIC_API_KEY env var)
            model: Model name to use (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            storage_path: Path where memories will be stored
            user_id: Unique identifier for the user
            embedding_model: Custom embedding model (defaults to LocalEmbeddingModel)
            salience_threshold: Threshold for memory worthiness (-0.1 to 0.2, default 0.0)
                              Lower = more permissive, Higher = more strict
            operation_mode: Operation mode - "local" (sentence-transformers),
                          "online" (OpenAI API), or "lightweight" (keywords only)
            scheduler_interval_seconds: Interval in seconds to check for due tasks (default: 60)
            curation_interval_seconds: Interval in seconds to run memory curation (default: 3600)
            **kwargs: Additional arguments passed to anthropic.Anthropic()
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.user_id = user_id
        self.storage_path = storage_path
        self.salience_threshold = salience_threshold
        self.operation_mode = operation_mode
        self._provided_embedding_model = embedding_model
        self.scheduler_interval_seconds = scheduler_interval_seconds
        self.curation_interval_seconds = curation_interval_seconds
        
        # Lazy-loaded attributes
        self._embedding_model = None
        self._vector_storage = None
        self._graph_storage = None
        self._salience_gate = None
        self._search_service = None
        self._consolidation_service = None
        
        # --- Curation Service Lifecycle ---
        self._curation_service = None
        self._scheduler_service = None
        
        # Observability
        self.last_trace: Optional["Trace"] = None
        
        # Initialize Claude client (lightweight, fast)
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key, **kwargs)
        else:
            self.client = anthropic.Anthropic(**kwargs)
        
        # Register the close method to be called upon script exit
        import atexit
        atexit.register(self.close)
        
        # Claude's tool definition
        self.tool_schema = [
            {
                "name": "search_memory",
                "description": "Searches the user's long-term memory to answer questions about past conversations or stored facts. Use this for any non-trivial question that requires recalling past information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A specific and detailed question or search query for the MemLayer."
                        },
                        "search_tier": {
                            "type": "string",
                            "enum": ["fast", "balanced", "deep"],
                            "description": "The desired depth of the search. 'fast' is for quick lookups (<100ms). 'balanced' is for more thorough searches (<500ms). 'deep' is for comprehensive, multi-step reasoning (<2s)."
                        }
                    },
                    "required": ["query", "search_tier"]
                }
            },
            {
                "name": "schedule_task",
                "description": "Schedules a task or reminder for the user at a future date and time. Use this when the user asks to be reminded about something.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "A detailed, self-contained description of the task to be done. Should include all necessary context."
                        },
                        "due_date": {
                            "type": "string",
                            "description": "The future date and time the task is due, preferably in ISO 8601 format (e.g., '2025-12-25T09:00:00'). The model should calculate this based on the user's request and the current date if necessary."
                        }
                    },
                    "required": ["task_description", "due_date"]
                }
            }
        ]
    
    @property
    def curation_service(self) -> "CurationService":
        """Lazy-load the curation service."""
        if self._curation_service is None:
            from ..services import CurationService
            self._curation_service = CurationService(
                self.vector_storage, 
                self.graph_storage,
                interval_seconds=self.curation_interval_seconds
            )
            self._curation_service.start()
        return self._curation_service
    
    @property
    def embedding_model(self) -> "BaseEmbeddingModel":
        """Lazy-load the embedding model only when needed. Returns None in LIGHTWEIGHT mode."""
        if self.operation_mode == "lightweight":
            return None  # LIGHTWEIGHT mode doesn't use embeddings
            
        if self._embedding_model is None:
            if self._provided_embedding_model is None:
                # Use appropriate embedding model based on mode
                if self.operation_mode == "online":
                    import os
                    from ..embedding_models import OpenAIEmbeddingModel
                    import openai
                    print("Initializing OpenAI embedding model (text-embedding-3-small)...")
                    # Create OpenAI client for embeddings
                    openai_key = os.getenv("OPENAI_API_KEY")
                    if not openai_key:
                        raise ValueError("ONLINE mode requires OPENAI_API_KEY environment variable")
                    openai_client = openai.OpenAI(api_key=openai_key)
                    self._embedding_model = OpenAIEmbeddingModel(
                        client=openai_client,
                        model_name="text-embedding-3-small"
                    )
                else:  # local mode
                    from ..embedding_models import LocalEmbeddingModel
                    print("Initializing local embedding model 'all-MiniLM-L6-v2'...")
                    self._embedding_model = LocalEmbeddingModel()
            else:
                self._embedding_model = self._provided_embedding_model
        return self._embedding_model
    
    @property
    def vector_storage(self) -> "ChromaStorage":
        """Lazy-load vector storage only when needed. Returns None in LIGHTWEIGHT mode."""
        if self.operation_mode == "lightweight":
            return None  # LIGHTWEIGHT mode uses graph-only storage
            
        if self._vector_storage is None:
            from ..storage.chroma import ChromaStorage
            self._vector_storage = ChromaStorage(self.storage_path, dimension=self.embedding_model.dimension)
        return self._vector_storage
    
    @property
    def graph_storage(self) -> "NetworkXStorage":
        """Lazy-load graph storage only when needed."""
        if self._graph_storage is None:
            from ..storage.networkx import NetworkXStorage
            self._graph_storage = NetworkXStorage(self.storage_path)
        return self._graph_storage
    
    @property
    def salience_gate(self) -> "SalienceGate":
        """Lazy-load salience gate only when needed."""
        if self._salience_gate is None:
            from ..ml_gate import SalienceGate, SalienceMode
            import os
            
            # Convert string mode to enum
            mode = SalienceMode(self.operation_mode.lower())
            
            # For LOCAL mode, share embedding model
            # For ONLINE mode, pass OpenAI API key
            self._salience_gate = SalienceGate(
                threshold=self.salience_threshold,
                embedding_model=self.embedding_model if mode == SalienceMode.LOCAL else None,
                mode=mode,
                openai_api_key=os.getenv("OPENAI_API_KEY") if mode == SalienceMode.ONLINE else None
            )
        return self._salience_gate
    
    @property
    def search_service(self) -> "SearchService":
        """Lazy-load search service only when needed."""
        if self._search_service is None:
            from ..services import SearchService
            self._search_service = SearchService(self.vector_storage, self.graph_storage, self.embedding_model)
        return self._search_service
    
    @property
    def consolidation_service(self) -> "ConsolidationService":
        """Lazy-load consolidation service only when needed."""
        if self._consolidation_service is None:
            from ..services import ConsolidationService
            self._consolidation_service = ConsolidationService(
                self.vector_storage,
                self.graph_storage,
                self.embedding_model,
                self.salience_gate,
                llm_client=self
            )
        return self._consolidation_service

    def chat(self, messages: list, **kwargs) -> str:
        """
        Send a chat completion request with memory capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments for the completion (will override defaults)
        
        Returns:
            str: The assistant's response
        """
        # Ensure curation service is started (accessing the property triggers lazy load + start)
        _ = self.curation_service
        
        self.last_trace = None  # Reset trace for each new chat call
        
        triggered_context = self.search_service.get_triggered_tasks_context(self.user_id)
        if triggered_context:
            # Prepend the task reminders as a system message to guide the LLM's response.
            # This ensures the LLM is aware of due tasks at the start of the turn.
            messages.insert(0, {"role": "user", "content": triggered_context})
        
        user_query = messages[-1]['content']
        
        # Apply defaults if not overridden
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": self.tool_schema,
            "tool_choice": {"type": "auto"},
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        completion_kwargs.update(kwargs)

        try:
            # 1. First call to Claude with the tool available
            response = self.client.messages.create(**completion_kwargs)
            
            # 2. Check if the model decided to use tools
            if response.stop_reason == "tool_use":
                # Claude can return multiple tool calls in a single response
                tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
                
                if not tool_use_blocks:
                    raise ValueError("Stop reason was 'tool_use' but no tool_use blocks were found.")

                # Append the assistant's response with tool calls
                messages.append({"role": "assistant", "content": response.content})
                
                # Process each tool call
                tool_results = []
                for tool_use_block in tool_use_blocks:
                    tool_name = tool_use_block.name
                    tool_input = tool_use_block.input
                    tool_call_id = tool_use_block.id

                    if tool_name == "search_memory":
                        # Execute the tool with graph traversal support
                        search_output = self.search_service.search(
                            query=tool_input.get("query", ""),
                            user_id=self.user_id,
                            search_tier=tool_input.get("search_tier", "balanced"),
                            llm_client=self  # Enable entity extraction for "deep" searches
                        )
                        search_result_text = search_output["result"]
                        self.last_trace = search_output["trace"]  # Store the trace object
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": search_result_text
                        })
                    
                    elif tool_name == "schedule_task":
                        try:
                            import dateutil.parser
                            description = tool_input.get("task_description")
                            due_date_str = tool_input.get("due_date")
                            
                            # Convert the date string to a timestamp
                            due_timestamp = dateutil.parser.parse(due_date_str).timestamp()
                            
                            # Call the new graph storage method
                            task_id = self.graph_storage.add_task(description, due_timestamp, self.user_id)
                            
                            tool_response = f"Task successfully scheduled with ID: {task_id}. I will remind you when it's due."
                        except ImportError:
                            print("Error: dateutil.parser is required for schedule_task. Install with: pip install python-dateutil")
                            tool_response = "Error: Missing required library for date parsing."
                        except Exception as e:
                            print(f"Error scheduling task: {e}")
                            tool_response = "Error: Could not schedule the task due to an invalid date format or other issue."
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_response
                        })
                    
                    else:
                        # Unknown tool
                        print(f"Warning: Claude called unknown tool '{tool_name}'")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": f"Error: Unknown tool '{tool_name}'."
                        })

                # Append all tool results as a single user message
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

                # Create new kwargs without conflicting keys
                second_kwargs = {k: v for k, v in completion_kwargs.items() if k not in ['tools', 'tool_choice']}
                second_kwargs['messages'] = messages
                
                second_response = self.client.messages.create(**second_kwargs)
                final_response = second_response.content[0].text
            else:
                # No tool call, just a direct text response
                final_response = response.content[0].text

        except Exception as e:
            print(f"An error occurred during Claude chat: {e}")
            final_response = "I'm sorry, an error occurred while processing your request."

        # 5. Consolidate the full interaction
        full_interaction = f"User: {user_query}\nAssistant: {final_response}"
        self.consolidation_service.consolidate(full_interaction, self.user_id)

        return final_response

    def analyze_and_extract_knowledge(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extracts facts, entities, and relationships from text for the knowledge graph.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict with keys 'facts', 'entities', and 'relationships'
        """
        system_prompt = """
You are a Knowledge Graph Engineer AI. Your task is to analyze text and deconstruct it into a structured knowledge graph.
You must identify:
1.  **facts**: A list of simple, atomic, self-contained statements.
2.  **entities**: A list of key nouns (people, places, projects, concepts). Each entity should have a 'name' and a 'type'.
3.  **relationships**: A list of connections between entities. Each relationship must have a 'subject' (entity name), a 'predicate' (the verb or connecting phrase), and an 'object' (entity name).

Respond ONLY with a valid JSON object with the keys "facts", "entities", and "relationships". Ensure all values in the 'subject' and 'object' fields of the relationships correspond to a 'name' from the entities list.

Example Input:
"John, the lead engineer for Project Phoenix, confirmed that the new server deployment in the London office is complete. This server's IP is 192.168.1.101."

Example JSON Output:
{
  "facts": [
    {"fact": "The new server deployment in the London office is complete."},
    {"fact": "The IP address of the new server in the London office is 192.168.1.101."}
  ],
  "entities": [
    {"name": "John", "type": "Person"},
    {"name": "Project Phoenix", "type": "Project"},
    {"name": "London office", "type": "Location"},
    {"name": "server deployment", "type": "Event"},
    {"name": "192.168.1.101", "type": "IP Address"}
  ],
  "relationships": [
    {"subject": "John", "predicate": "is lead engineer for", "object": "Project Phoenix"},
    {"subject": "John", "predicate": "confirmed completion of", "object": "server deployment"},
    {"subject": "server deployment", "predicate": "is located in", "object": "London office"}
  ]
}

Do not include any other text, explanations, or apologies. Only the JSON object.
"""
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
                max_tokens=2048,
                temperature=0.0
            )
            
            content = response.content[0].text
            if not content:
                return {"facts": [], "entities": [], "relationships": []}

            # Claude might sometimes wrap the JSON in ```json ... ```, so we strip it
            if content.strip().startswith("```json"):
                content = content.strip()[7:-3]
            elif content.strip().startswith("```"):
                content = content.strip()[3:-3]

            knowledge_graph = json.loads(content)
            
            # Basic validation to ensure keys exist
            knowledge_graph.setdefault("facts", [])
            knowledge_graph.setdefault("entities", [])
            knowledge_graph.setdefault("relationships", [])
            
            return knowledge_graph
        except Exception as e:
            print(f"An unexpected error occurred during Claude knowledge extraction: {e}")
            # Fallback to a simple fact to ensure something is saved
            return {"facts": [{"fact": text}], "entities": [], "relationships": []}

    def update_from_text(self, text_block: str):
        """
        Directly ingests a block of text into the memory bank.

        This method is the most efficient way to add external knowledge (e.g., from
        documents, emails, or other sources) to the user's memory. It bypasses
        the conversational chat loop and directly engages the consolidation service.

        Args:
            text_block (str): The text content to be analyzed and saved to memory.
        """
        print(f"Updating memory for user '{self.user_id}' from text block...")
        # The consolidation service is already designed to run in the background,
        # so we can simply call it directly.
        self.consolidation_service.consolidate(text_block, self.user_id)
        print("-> Knowledge extraction and consolidation initiated in the background.")

    def synthesize_answer(self, question: str, return_object: bool = False):
        """
        Provides a high-quality, memory-grounded answer to a specific question.

        This method encapsulates the entire cognitive loop for question-answering:
        1. Performs a "deep" hybrid search (vector + graph) for relevant context.
        2. Constructs a highly-optimized prompt for the LLM, forcing it to use
           only the provided context.
        3. Generates a synthesized answer.
        4. Returns the answer and optional metadata about the sources.

        Args:
            question (str): The user's question.
            return_object (bool): If True, returns a detailed AnswerObject. 
                                  If False (default), returns only the answer text.

        Returns:
            str | AnswerObject: The synthesized answer.
        """
        print(f"Synthesizing answer for question: '{question}'")
        
        # --- Step 1: Perform a guaranteed "deep" search ---
        # We force the deep tier to get the richest possible context.
        search_output = self.search_service.search(
            query=question,
            user_id=self.user_id,
            search_tier="deep",
            llm_client=self
        )
        context = search_output["result"]
        self.last_trace = search_output["trace"] # Store the trace

        # --- Step 2: Construct the Synthesis Prompt ---
        # This prompt is engineered to prevent hallucination and force grounding.
        synthesis_prompt = f"""
You are a synthesis model. Your task is to answer the user's question based *only* on the context provided below. Do not use any prior knowledge. If the context does not contain the answer, state that the information is not available in the memory.

**CONTEXT:**
---
{context}
---

**QUESTION:**
{question}

**Synthesized Answer:**
"""
        
        # --- Step 3: Generate the Final Answer ---
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0, # Low temperature for factual, grounded answers
            )
            answer_text = response.content[0].text
        except Exception as e:
            print(f"Error during synthesis LLM call: {e}")
            answer_text = "Sorry, I encountered an error while synthesizing the answer."

        # --- Step 4: Return the result in the desired format ---
        if return_object:
            from ..observability import AnswerObject # Define this new Pydantic model
            return AnswerObject(
                question=question,
                answer=answer_text,
                context=context,
                trace=self.last_trace
            )
        else:
            return answer_text

    def extract_query_entities(self, query: str) -> List[str]:
        """
        Extracts key entities from a search query for graph traversal.
        
        Args:
            query: The search query
            
        Returns:
            List of entity names found in the query
        """
        prompt = f"""Extract the key entities (people, places, projects, concepts) from this query. 
Return ONLY a JSON array of entity names, nothing else.

Query: {query}

JSON array:"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0
            )
            
            content = response.content[0].text.strip()
            
            # Clean up potential markdown code blocks
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            entities = json.loads(content)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            print(f"Error extracting query entities: {e}")
            return []

    def close(self):
        """Release resources and close storage connections."""
        try:
            if self._curation_service is not None:
                self._curation_service.stop()
            if self._vector_storage is not None:
                self._vector_storage.close()
            if self._graph_storage is not None:
                self._graph_storage.close()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

