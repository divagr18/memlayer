from typing import Dict, List, Optional, TYPE_CHECKING
import openai
import json
import dateutil.parser

# Use TYPE_CHECKING to avoid slow imports at module load time
if TYPE_CHECKING:
    from ..ml_gate import SalienceGate
    from ..storage.chroma import ChromaStorage
    from ..storage.networkx import NetworkXStorage
    from ..storage.memgraph import MemgraphStorage
    from ..services import SearchService, ConsolidationService
    from ..embedding_models import BaseEmbeddingModel, LocalEmbeddingModel
    from ..observability import Trace
    from .base import BaseLLMWrapper
else:
    # Import these at runtime when actually needed
    SalienceGate = None
    ChromaStorage = None
    NetworkXStorage = None
    MemgraphStorage = None
    SearchService = None
    ConsolidationService = None
    BaseEmbeddingModel = None
    LocalEmbeddingModel = None
    Trace = None
    BaseLLMWrapper = object  # Use object as base for now


class OpenAI(BaseLLMWrapper):
    """
    A memory-enhanced OpenAI client that can be used standalone.
    
    Usage:
        from memory_bank.wrappers.openai import OpenAI
        
        client = OpenAI(
            api_key="your-api-key",
            model="gpt-4",
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
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        storage_path: str = "./memory_bank_data",
        user_id: str = "default_user",
        embedding_model: Optional["BaseEmbeddingModel"] = None,
        salience_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize a memory-enhanced OpenAI client.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: Model name to use (e.g., "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo")
            temperature: Sampling temperature (0.0 to 2.0)
            storage_path: Path where memories will be stored
            user_id: Unique identifier for the user
            embedding_model: Custom embedding model (defaults to LocalEmbeddingModel)
            salience_threshold: Threshold for memory worthiness (-0.1 to 0.2, default 0.0)
                              Lower = more permissive, Higher = more strict
            **kwargs: Additional arguments passed to openai.OpenAI()
        """
        self.model = model
        self.temperature = temperature
        self.user_id = user_id
        self.storage_path = storage_path
        self.salience_threshold = salience_threshold
        self._provided_embedding_model = embedding_model
        
        # Lazy-loaded attributes
        self._embedding_model = None
        self._vector_storage = None
        self._graph_storage = None
        self._salience_gate = None
        self._search_service = None
        self._consolidation_service = None
        
        # Initialize OpenAI client (lightweight, fast)
        if api_key:
            self.client = openai.OpenAI(api_key=api_key, **kwargs)
        else:
            self.client = openai.OpenAI(**kwargs)
        
        # Set up memory search tool schema (no loading required)
        self.tool_schema = [{
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Searches the user's long-term memory. Use this for any non-trivial question that requires recalling past information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A specific and detailed question or search query for the memory bank."
                        },
                        "search_tier": {
                            "type": "string",
                            "enum": ["fast", "balanced", "deep"],
                            "description": "The desired depth of the search. 'fast' is for quick lookups (<100ms). 'balanced' is for more thorough searches (<500ms). 'deep' is for comprehensive, multi-step reasoning (<2s)."
                        }
                    },
                    "required": ["query", "search_tier"]
                }
            }
        },
            {
                "type": "function",
                "function": {
                    "name": "schedule_task",
                    "description": "Schedules a task or reminder for the user at a future date and time. Use this when the user asks to be reminded about something.",
                    "parameters": {
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
            }
        ]
        self.last_trace: Optional["Trace"] = None
    
    @property
    def embedding_model(self) -> "BaseEmbeddingModel":
        """Lazy-load the embedding model only when needed."""
        if self._embedding_model is None:
            from ..embedding_models import LocalEmbeddingModel
            if self._provided_embedding_model is None:
                print("Initializing local embedding model 'all-MiniLM-L6-v2'...")
                self._embedding_model = LocalEmbeddingModel()
            else:
                self._embedding_model = self._provided_embedding_model
        return self._embedding_model
    
    @property
    def vector_storage(self) -> "ChromaStorage":
        """Lazy-load vector storage only when needed."""
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
            from ..ml_gate import SalienceGate
            self._salience_gate = SalienceGate(threshold=self.salience_threshold)
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
    
    def chat(self, messages: List[Dict[str, str]], **kwargs):
        """
        Send a chat completion request with memory capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments for the completion (will override defaults)
        
        Returns:
            str: The assistant's response
        """
        triggered_context = self.search_service.get_triggered_tasks_context(self.user_id)
        if triggered_context:
            # Prepend the task reminders as a system message to guide the LLM's response.
            # This ensures the LLM is aware of due tasks at the start of the turn.
            messages.insert(0, {"role": "system", "content": triggered_context})

        # Apply defaults if not overridden
        completion_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "tools": [self.tool_schema],
            "tool_choice": "auto",
        }
        completion_kwargs.update(kwargs)
        
        self.last_trace = None  # Reset trace for each new chat call
        
        # 1. Make the first call to the LLM with the memory tool available
        try:
            response = self.client.chat.completions.create(**completion_kwargs)
            response_message = response.choices[0].message
        except Exception as e:
            print(f"Error during initial LLM call: {e}")
            return "Sorry, I encountered an error trying to process your request."

        # 2. Check if the LLM decided to use our tool
        if not response_message.tool_calls:
            # No tool call, this is the "fast path" for simple conversation.
            final_response = response_message.content
        else:
            # --- HANDLE MULTIPLE TOOL CALLS ---
            # The LLM might call multiple tools in one turn. We need to handle this.
            messages.append(response_message)  # Append assistant's turn with tool calls
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                
                if function_name == "search_memory":
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        search_tier = function_args.get("search_tier", "balanced")
                        
                        # 3. Execute the fully-traced search via the SearchService
                        # Pass self as llm_client to enable deep search with graph traversal
                        search_output = self.search_service.search(
                            query=query, 
                            user_id=self.user_id, 
                            search_tier=search_tier,
                            llm_client=self  # Enable entity extraction for "deep" searches
                        )
                        search_result_text = search_output["result"]
                        self.last_trace = search_output["trace"]  # Store the trace object

                        # Append the tool result message
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": search_result_text,
                        })

                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError in search_memory tool call: {e}")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": "Error: Failed to parse tool arguments.",
                        })
                    except Exception as e:
                        print(f"Error during search_memory tool execution: {e}")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": "Sorry, I encountered an error while searching my memory.",
                        })
                        if self.last_trace:
                            self.last_trace.conclude(error=e)
                
                elif function_name == "schedule_task":
                    try:
                        import dateutil.parser
                        function_args = json.loads(tool_call.function.arguments)
                        description = function_args.get("task_description")
                        due_date_str = function_args.get("due_date")
                        
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

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_response,
                    })
                
                else:
                    # Unknown tool - return error message
                    print(f"Warning: LLM called unknown tool '{function_name}'")
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error: Unknown tool '{function_name}'.",
                    })

            # After processing all tool calls, send the results back to the LLM for final response
            try:
                # Create new kwargs without conflicting keys
                second_kwargs = {k: v for k, v in completion_kwargs.items() if k not in ['tools', 'tool_choice']}
                second_kwargs['messages'] = messages
                
                second_response = self.client.chat.completions.create(**second_kwargs)
                final_response = second_response.choices[0].message.content
            except Exception as e:
                print(f"Error during second LLM call after tool execution: {e}")
                final_response = "Sorry, I encountered an error while processing the tool results."

        # 5. Consolidate the full interaction in the background
        user_query = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""
        full_interaction = f"User: {user_query}\nAssistant: {final_response}"
        self.consolidation_service.consolidate(full_interaction, self.user_id)

        return final_response
    
    def analyze_and_extract_knowledge(self, text: str) -> Dict:
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
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if not content:
                return {"facts": [], "entities": [], "relationships": []}
            
            knowledge_graph = json.loads(content)
            
            # Basic validation to ensure keys exist
            knowledge_graph.setdefault("facts", [])
            knowledge_graph.setdefault("entities", [])
            knowledge_graph.setdefault("relationships", [])
            
            return knowledge_graph
        except Exception as e:
            print(f"An unexpected error occurred during knowledge extraction: {e}")
            # Fallback to a simple fact to ensure something is saved
            return {"facts": [{"fact": text}], "entities": [], "relationships": []}
    def extract_query_entities(self, query: str) -> List[str]:
        """
        Uses a focused LLM call to identify key entities in a search query.
        """
        system_prompt = """
You are an efficient entity extraction model. Your task is to identify the main nouns or concepts in the user's query that could be looked up in a knowledge graph.
Do not explain. Do not use conversational filler. Respond ONLY with a valid JSON object containing a single key "entities", which is a list of the extracted entity strings.
If no specific entities are found, return an empty list.

Example 1:
Query: "Who is the lead engineer for Project Phoenix?"
Your JSON Output:
{
  "entities": ["Project Phoenix"]
}

Example 2:
Query: "What was the flight number for my trip to Tokyo?"
Your JSON Output:
{
  "entities": ["Tokyo"]
}

Example 3:
Query: "Tell me about John."
Your JSON Output:
{
  "entities": ["John"]
}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model, # Use the configured model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if not content:
                return []
            
            data = json.loads(content)
            entities = data.get("entities", [])
            
            # Ensure it's a list of strings
            if isinstance(entities, list) and all(isinstance(e, str) for e in entities):
                return entities
            return []
            
        except Exception as e:
            print(f"An error occurred during query entity extraction: {e}")
            return []
