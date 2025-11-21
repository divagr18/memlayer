from google import genai
from google.genai import types
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


class Gemini(BaseLLMWrapper):
    """
    A memory-enhanced Google Gemini client that can be used standalone.
    
    Usage:
        from memlayer.wrappers.gemini import Gemini
        
        client = Gemini(
            api_key="your-api-key",
            model="gemini-2.5-flash-lite",
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
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7,
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
        Initialize a memory-enhanced Gemini client.
        
        Args:
            api_key: Google API key (if None, will use GOOGLE_API_KEY env var)
            model: Model name to use (e.g., "gemini-2.5-flash-lite", "gemini-2.5-pro")
            temperature: Sampling temperature (0.0 to 1.0)
            storage_path: Path where memories will be stored
            user_id: Unique identifier for the user
            embedding_model: Custom embedding model (defaults to LocalEmbeddingModel)
            salience_threshold: Threshold for memory worthiness (-0.1 to 0.2, default 0.0)
                              Lower = more permissive, Higher = more strict
            operation_mode: Operation mode - "local" (sentence-transformers),
                          "online" (OpenAI API), or "lightweight" (keywords only)
            scheduler_interval_seconds: Interval in seconds to check for due tasks (default: 60)
            curation_interval_seconds: Interval in seconds to run memory curation (default: 3600)
            **kwargs: Additional arguments passed to genai.Client()
        """
        self.model = model
        self.temperature = temperature
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
        
        # Initialize Gemini client (lightweight, fast)
        if api_key:
            self.client = genai.Client(api_key=api_key, **kwargs)
        else:
            self.client = genai.Client(**kwargs)
        
        # Register the close method to be called upon script exit
        import atexit
        atexit.register(self.close)
        
        # Define the tool schema using the google-genai SDK format
        self.tool_schema = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search_memory",
                    description="Searches the user's long-term memory to answer questions about past conversations or stored facts. Use this for any non-trivial question that requires recalling past information.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "query": types.Schema(
                                type=types.Type.STRING,
                                description="A specific and detailed question or search query for the Memlayer."
                            ),
                            "search_tier": types.Schema(
                                type=types.Type.STRING,
                                enum=["fast", "balanced", "deep"],
                                description="The desired depth of the search. 'fast' is for quick lookups, 'balanced' is for thorough searches, 'deep' is for comprehensive reasoning."
                            )
                        },
                        required=["query", "search_tier"]
                    )
                ),
                types.FunctionDeclaration(
                    name="schedule_task",
                    description="Schedules a task or reminder for the user at a future date and time. Use this when the user asks to be reminded about something.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "task_description": types.Schema(
                                type=types.Type.STRING,
                                description="A detailed, self-contained description of the task to be done. Should include all necessary context."
                            ),
                            "due_date": types.Schema(
                                type=types.Type.STRING,
                                description="The future date and time the task is due, preferably in ISO 8601 format (e.g., '2025-12-25T09:00:00'). The model should calculate this based on the user's request and the current date if necessary."
                            )
                        },
                        required=["task_description", "due_date"]
                    )
                )
            ]
        )
    
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

    def chat(self, messages: list, stream: bool = False, **kwargs) -> str:
        """
        Send a chat completion request with memory capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stream: If True, returns a generator that yields response chunks
            **kwargs: Additional arguments for the completion
        
        Returns:
            str | Generator: The assistant's response (string if stream=False, generator if stream=True)
        """
        # Ensure curation service is started (accessing the property triggers lazy load + start)
        _ = self.curation_service
        
        self.last_trace = None  # Reset trace for each new chat call
        
        triggered_context = self.search_service.get_triggered_tasks_context(self.user_id)
        if triggered_context:
            # Prepend the task reminders as a system message to guide the LLM's response.
            # This ensures the LLM is aware of due tasks at the start of the turn.
            messages.insert(0, {"role": "user", "content": triggered_context})
        
        # Convert messages to Gemini's content format
        contents = self._prepare_contents(messages)
        user_query = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""
        
        # Handle streaming mode
        if stream:
            return self._stream_chat(contents, user_query, kwargs)

        try:
            # 1. First call to Gemini with the tool available
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    tools=[self.tool_schema],
                    **kwargs
                )
            )
            
            # 2. Check if the model made function calls
            function_calls = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
            
            if function_calls:
                # 3. Process all function calls
                # Append the model's response with function calls
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(function_call=fc) for fc in function_calls]
                ))
                
                # Process each function call and collect responses
                function_responses = []
                for function_call in function_calls:
                    function_name = function_call.name
                    args = function_call.args
                    
                    if function_name == "search_memory":
                        # Execute the memory search with graph traversal support
                        search_output = self.search_service.search(
                            query=args.get("query", ""),
                            user_id=self.user_id,
                            search_tier=args.get("search_tier", "balanced"),
                            llm_client=self  # Enable entity extraction for "deep" searches
                        )
                        search_result_text = search_output["result"]
                        self.last_trace = search_output["trace"]  # Store the trace object
                        
                        function_responses.append(types.Part(
                            function_response=types.FunctionResponse(
                                name="search_memory",
                                response={"content": search_result_text}
                            )
                        ))
                    
                    elif function_name == "schedule_task":
                        try:
                            import dateutil.parser
                            description = args.get("task_description")
                            due_date_str = args.get("due_date")
                            
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
                        
                        function_responses.append(types.Part(
                            function_response=types.FunctionResponse(
                                name="schedule_task",
                                response={"content": tool_response}
                            )
                        ))
                    
                    else:
                        # Unknown function
                        print(f"Warning: Gemini called unknown function '{function_name}'")
                        function_responses.append(types.Part(
                            function_response=types.FunctionResponse(
                                name=function_name,
                                response={"content": f"Error: Unknown function '{function_name}'."}
                            )
                        ))
                
                # 4. Append all function responses
                contents.append(types.Content(
                    role="user",  # Function responses go as user role in new SDK
                    parts=function_responses
                ))
                
                # Second call to get the final response
                config_kwargs = {k: v for k, v in kwargs.items() if k not in ['tools']}
                second_response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=self.temperature, **config_kwargs)
                )
                final_response = second_response.text
            else:
                # No function call, return direct response
                final_response = response.text

        except Exception as e:
            print(f"An error occurred during Gemini chat: {e}")
            import traceback
            traceback.print_exc()
            final_response = "I'm sorry, an error occurred while processing your request."

        # 5. Consolidate the full interaction in background
        full_interaction = f"User: {user_query}\nAssistant: {final_response}"
        self.consolidation_service.consolidate(full_interaction, self.user_id)

        return final_response
    
    def _stream_chat(self, contents: list, user_query: str, config_kwargs: dict):
        """
        Helper method to handle streaming responses for Gemini.
        
        Args:
            contents: Prepared contents for Gemini
            user_query: The user's query for consolidation
            config_kwargs: Configuration arguments
            
        Yields:
            str: Response chunks from Gemini
        """
        try:
            stream_response = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    tools=[self.tool_schema],
                    **config_kwargs
                )
            )
            
            full_response = ""
            function_calls = []
            
            for chunk in stream_response:
                # Handle text responses
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
                
                # Handle function calls
                if chunk.candidates and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)
            
            # If there were function calls, handle them
            if function_calls:
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(function_call=fc) for fc in function_calls]
                ))
                
                function_responses = []
                for function_call in function_calls:
                    function_name = function_call.name
                    
                    if function_name == "search_memory":
                        try:
                            query = function_call.args.get("query")
                            search_tier = function_call.args.get("search_tier", "balanced")
                            
                            search_output = self.search_service.search(
                                query=query,
                                user_id=self.user_id,
                                search_tier=search_tier,
                                llm_client=self
                            )
                            search_result_text = search_output["result"]
                            self.last_trace = search_output["trace"]
                            
                            function_responses.append(types.Part(
                                function_response=types.FunctionResponse(
                                    name=function_name,
                                    response={"result": search_result_text}
                                )
                            ))
                        except Exception as e:
                            print(f"Error during search_memory: {e}")
                            function_responses.append(types.Part(
                                function_response=types.FunctionResponse(
                                    name=function_name,
                                    response={"result": "Error searching memory."}
                                )
                            ))
                    
                    elif function_name == "schedule_task":
                        try:
                            import dateutil.parser
                            description = function_call.args.get("task_description")
                            due_date_str = function_call.args.get("due_date")
                            due_timestamp = dateutil.parser.parse(due_date_str).timestamp()
                            task_id = self.graph_storage.add_task(description, due_timestamp, self.user_id)
                            tool_response = f"Task successfully scheduled with ID: {task_id}."
                        except Exception as e:
                            print(f"Error scheduling task: {e}")
                            tool_response = "Error: Could not schedule the task."
                        
                        function_responses.append(types.Part(
                            function_response=types.FunctionResponse(
                                name=function_name,
                                response={"result": tool_response}
                            )
                        ))
                
                contents.append(types.Content(role="user", parts=function_responses))
                
                # Get final response after tool execution
                second_stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        **config_kwargs
                    )
                )
                
                for second_chunk in second_stream:
                    if second_chunk.text:
                        yield second_chunk.text
            
            # Consolidate after streaming completes
            full_interaction = f"User: {user_query}\nAssistant: {full_response}"
            self.consolidation_service.consolidate(full_interaction, self.user_id)
                        
        except Exception as e:
            print(f"Error during streaming: {e}")
            import traceback
            traceback.print_exc()
            yield "Sorry, I encountered an error while streaming the response."

    def analyze_and_extract_knowledge(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extracts facts, entities, and relationships from text for the knowledge graph.
        Uses a fast model (gemini-2.5-flash-lite) for efficient extraction.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict with keys 'facts', 'entities', and 'relationships'
        """
        from datetime import datetime
        
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p %Z")
        
        # Use fast flash model for extraction instead of the main model
        extraction_model = "gemini-2.5-flash-lite"
        
        system_prompt = f"""
You are a Knowledge Graph Engineer AI. Your task is to analyze text and deconstruct it into a structured knowledge graph.
The current date and time is {current_datetime}.
You must identify:
1.  **facts**: A list of simple, atomic statements. For each fact, assign an 'importance_score' (float 0.1-1.0) and an 'expiration_date' (ISO 8601 string or null if it doesn't expire).
2.  **entities**: A list of key nouns (people, places, projects, concepts). Each entity should have a 'name' and a 'type'.
3.  **relationships**: A list of connections between entities. Each relationship must have a 'subject' (entity name), a 'predicate' (the verb or connecting phrase), and an 'object' (entity name).

Respond ONLY with a valid JSON object with the keys "facts", "entities", and "relationships". Ensure all values in the 'subject' and 'object' fields of the relationships correspond to a 'name' from the entities list.

Example Input:
"John confirmed the temporary door code is 1234 for the next 24 hours. This is for Project Phoenix, which is our top priority."

Example JSON Output:
{{
  "facts": [
    {{"fact": "The temporary door code is 1234.", "importance_score": 0.8, "expiration_date": "2025-11-18T14:30:00Z"}},
    {{"fact": "Project Phoenix is the team's top priority.", "importance_score": 1.0, "expiration_date": null}}
  ],
  "entities": [
    {{"name": "John", "type": "Person"}},
    {{"name": "Project Phoenix", "type": "Project"}}
  ],
  "relationships": [
    {{"subject": "John", "predicate": "works on", "object": "Project Phoenix"}}
  ]
}}
"""
        prompt = f"{system_prompt}\n\nInput Text:\n{text}\n\nYour JSON Output:"
        
        try:
            response = self.client.models.generate_content(
                model=extraction_model,  # Use fast model, not self.model
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"  # Request JSON output
                )
            )
            content = response.text
            if not content:
                return {"facts": [], "entities": [], "relationships": []}
            
            knowledge_graph = json.loads(content)
            
            # Basic validation to ensure keys exist
            knowledge_graph.setdefault("facts", [])
            knowledge_graph.setdefault("entities", [])
            knowledge_graph.setdefault("relationships", [])
            
            return knowledge_graph
        except Exception as e:
            print(f"An unexpected error occurred during Gemini knowledge extraction: {e}")
            import traceback
            traceback.print_exc()
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
            response = self.client.models.generate_content(
                model=self.model,
                contents=synthesis_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, # Low temperature for factual, grounded answers
                )
            )
            answer_text = response.text
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
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            content = response.text.strip()
            entities = json.loads(content)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            print(f"Error extracting query entities: {e}")
            return []

    def _prepare_contents(self, messages: list) -> list:
        """
        Converts a standard message list to Gemini's Content format using google-genai types.
        """
        contents = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg['content'])]
            ))
        return contents

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


# Legacy alias for backward compatibility
GeminiWrapper = Gemini

