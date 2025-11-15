import requests
import json
from typing import Dict, List, Optional, TYPE_CHECKING

# Use TYPE_CHECKING to avoid slow imports at module load time
if TYPE_CHECKING:
    from ..storage.chroma import ChromaStorage
    from ..storage.networkx import NetworkXStorage
    from ..storage.memgraph import MemgraphStorage
    from ..embedding_models import BaseEmbeddingModel, LocalEmbeddingModel
    from ..ml_gate import SalienceGate
    from ..services import SearchService, ConsolidationService
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
    BaseLLMWrapper = object


class Ollama(BaseLLMWrapper):
    """
    A memory-enhanced Ollama client that can be used standalone with local LLMs.
    
    Usage:
        from memory_bank.wrappers.ollama import Ollama
        
        client = Ollama(
            host="http://localhost:11434",
            model="qwen3:1.7b",
            storage_path="./my_memories",
            user_id="user_123"
        )
        
        response = client.chat(messages=[
            {"role": "user", "content": "What's my favorite color?"}
        ])
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:1.7b",
        temperature: float = 0.7,
        storage_path: str = "./memory_bank_data",
        user_id: str = "default_user",
        embedding_model: Optional["BaseEmbeddingModel"] = None,
        salience_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize a memory-enhanced Ollama client.
        
        Args:
            host: Ollama server URL (default: "http://localhost:11434")
            model: Model name to use (e.g., "qwen3:1.7b", "llama3:8b", "mistral:7b")
            temperature: Sampling temperature (0.0 to 1.0)
            storage_path: Path where memories will be stored
            user_id: Unique identifier for the user
            embedding_model: Custom embedding model (defaults to LocalEmbeddingModel)
            salience_threshold: Threshold for memory worthiness (-0.1 to 0.2, default 0.0)
                              Lower = more permissive, Higher = more strict
            **kwargs: Additional arguments (currently unused)
        """
        self.host = host
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
        
        # Tool schema for prompting the model (Ollama uses simpler JSON format)
        self.tool_schema = [
            {
                "name": "search_memory",
                "description": "Searches the user's long-term memory to answer questions about past conversations or stored facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A specific and detailed question or search query."},
                        "search_tier": {"type": "string", "enum": ["fast", "balanced", "deep"], "description": "The depth of the search."}
                    },
                    "required": ["query", "search_tier"]
                }
            },
            {
                "name": "schedule_task",
                "description": "Schedules a task or reminder for the user at a future date and time. Use this when the user asks to be reminded about something.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string", "description": "A detailed, self-contained description of the task."},
                        "due_date": {"type": "string", "description": "The future date and time in ISO 8601 format (e.g., '2025-12-25T09:00:00')."}
                    },
                    "required": ["task_description", "due_date"]
                }
            }
        ]
    
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

    def _generate_tool_prompt(self, messages: list) -> str:
        """Creates a prompt that instructs the Ollama model to use our tools."""
        user_query = messages[-1]['content']
        tool_json = json.dumps(self.tool_schema, indent=2)
        
        return f"""
You have access to multiple tools. To use a tool, respond with a JSON object that matches one of the following schemas:
{tool_json}

If the user's query is simple (like a greeting), respond directly. Otherwise, use the appropriate tool.

User query: "{user_query}"
Your JSON response (or direct answer):
"""

    def chat(self, messages: list, **kwargs) -> str:
        """
        Send a chat completion request with memory capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments for the completion
        
        Returns:
            str: The assistant's response
        """
        triggered_context = self.search_service.get_triggered_tasks_context(self.user_id)
        if triggered_context:
            # Prepend the task reminders to guide the LLM's response.
            # This ensures the LLM is aware of due tasks at the start of the turn.
            messages.insert(0, {"role": "system", "content": triggered_context})
        
        # 1. First call to the LLM with the tool-use prompt
        tool_prompt = self._generate_tool_prompt(messages)
        response_text = self._call_ollama(tool_prompt, **kwargs)

        # 2. Check if the model's response is a tool call (a valid JSON)
        try:
            tool_call_data = json.loads(response_text)
            tool_name = tool_call_data.get("name")
            
            if tool_name == "search_memory":
                # It's a search_memory tool call!
                params = tool_call_data.get("parameters", {})
                query = params.get("query")
                search_tier = params.get("search_tier", "balanced")
                
                # 3. Execute the tool with graph traversal support
                search_output = self.search_service.search(
                    query, 
                    self.user_id, 
                    search_tier,
                    llm_client=self  # Enable entity extraction for "deep" searches
                )
                search_result_text = search_output["result"]
                
                # 4. Send the result back to the LLM for the final answer
                final_prompt = f"Based on the following information, please answer the user's original query.\n\nInformation:\n{search_result_text}\n\nUser Query: {messages[-1]['content']}"
                final_response = self._call_ollama(final_prompt, **kwargs)
            
            elif tool_name == "schedule_task":
                # It's a schedule_task tool call!
                try:
                    import dateutil.parser
                    params = tool_call_data.get("parameters", {})
                    description = params.get("task_description")
                    due_date_str = params.get("due_date")
                    
                    # Convert the date string to a timestamp
                    due_timestamp = dateutil.parser.parse(due_date_str).timestamp()
                    
                    # Call the graph storage method
                    task_id = self.graph_storage.add_task(description, due_timestamp, self.user_id)
                    
                    tool_response = f"Task successfully scheduled with ID: {task_id}. I will remind you when it's due."
                except ImportError:
                    print("Error: dateutil.parser is required for schedule_task. Install with: pip install python-dateutil")
                    tool_response = "Error: Missing required library for date parsing."
                except Exception as e:
                    print(f"Error scheduling task: {e}")
                    tool_response = "Error: Could not schedule the task due to an invalid date format or other issue."
                
                # Send the result back to LLM for acknowledgement
                final_prompt = f"Please acknowledge to the user: {tool_response}"
                final_response = self._call_ollama(final_prompt, **kwargs)
            
            else:
                # It's JSON, but not a valid tool call
                final_response = response_text
        except json.JSONDecodeError:
            # Not a JSON response, so it's a direct answer
            final_response = response_text

        # 5. Consolidate in the background
        user_query = messages[-1]['content']
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
"""
        prompt = f"{system_prompt}\n\nInput Text:\n{text}\n\nYour JSON Output:"
        
        try:
            response_text = self._call_ollama(prompt)
            
            # Try to extract JSON from the response (in case there's extra text)
            if "```json" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                response_text = response_text[json_start:json_end]
            
            knowledge_graph = json.loads(response_text)
            
            # Basic validation to ensure keys exist
            knowledge_graph.setdefault("facts", [])
            knowledge_graph.setdefault("entities", [])
            knowledge_graph.setdefault("relationships", [])
            
            return knowledge_graph
        except Exception as e:
            print(f"An unexpected error occurred during Ollama knowledge extraction: {e}")
            # Fallback to a simple fact to ensure something is saved
            return {"facts": [{"fact": text}], "entities": [], "relationships": []}

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
            response_text = self._call_ollama(prompt)
            
            # Try to extract JSON from the response
            if "```json" in response_text or "```" in response_text:
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
            
            entities = json.loads(response_text)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            print(f"Error extracting query entities: {e}")
            return []

    def _call_ollama(self, prompt: str, **kwargs) -> str:
        """Helper function to call the Ollama API."""
        try:
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }
            # Allow overriding options
            if kwargs:
                request_data["options"].update(kwargs)
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=request_data
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return "Error: Could not connect to the local LLM."

