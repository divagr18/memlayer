from typing import Any, Dict, List, Optional, TYPE_CHECKING
import openai
import json
import re
import os
from ..config import is_debug_mode

# Use TYPE_CHECKING to avoid slow imports at module load time
if TYPE_CHECKING:
    from ..ml_gate import SalienceGate
    from ..storage.chroma import ChromaStorage
    from ..storage.networkx import NetworkXStorage
    from ..storage.memgraph import MemgraphStorage
    from ..services import SearchService, ConsolidationService, CurationService
    from ..embedding_models import BaseEmbeddingModel, LocalEmbeddingModel
    from ..observability import Trace
    from .base import BaseLLMWrapper
else:
    SalienceGate = None
    ChromaStorage = None
    NetworkXStorage = None
    MemgraphStorage = None
    SearchService = None
    ConsolidationService = None
    BaseEmbeddingModel = None
    LocalEmbeddingModel = None
    CurationService = None
    Trace = None
    BaseLLMWrapper = object


class Novita(BaseLLMWrapper):
    """
    A memory-enhanced client for Novita.ai (OpenAI-compatible API).

    Novita provides access to multiple open-source models through a unified,
    reliable API with great pricing.

    Usage:
        from memlayer.wrappers.novita import Novita

        client = Novita(
            api_key="your-api-key",
            model="deepseek/deepseek-v3.2",
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
        model: str = "deepseek/deepseek-v3.2",
        temperature: float = 0.7,
        storage_path: str = "./memlayer_data",
        user_id: str = "default_user",
        embedding_model: Optional["BaseEmbeddingModel"] = None,
        salience_threshold: float = 0.0,
        operation_mode: str = "online",
        scheduler_interval_seconds: int = 60,
        curation_interval_seconds: int = 3600,
        **kwargs,
    ):
        """
        Initialize a memory-enhanced Novita client.

        Args:
            api_key: Novita API key (if None, will use NOVITA_API_KEY env var)
            model: Model name to use (e.g., "deepseek/deepseek-v3.2", "zai-org/glm-5", "minimax/minimax-m2.5")
            temperature: Sampling temperature (0.0 to 2.0)
            storage_path: Path where memories will be stored
            user_id: Unique identifier for the user
            embedding_model: Custom embedding model (defaults to LocalEmbeddingModel)
            salience_threshold: Threshold for memory worthiness (-0.1 to 0.2, default 0.0)
                              Lower = more permissive, Higher = more strict
            operation_mode: Operation mode - "local" (sentence-transformers),
                          "online" (OpenAI embeddings API), or "lightweight" (keywords only)
            scheduler_interval_seconds: Interval in seconds to check for due tasks (default: 60)
            curation_interval_seconds: Interval in seconds to run memory curation (default: 3600)
            **kwargs: Additional arguments passed to openai.OpenAI()
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

        # Initialize OpenAI client pointing to Novita endpoint
        if api_key:
            self.client = openai.OpenAI(
                api_key=api_key, base_url="https://api.novita.ai/openai/v1", **kwargs
            )
        else:
            self.client = openai.OpenAI(
                base_url="https://api.novita.ai/openai/v1", **kwargs
            )

        # Register the close method to be called upon script exit
        import atexit

        atexit.register(self.close)

        # Tool Definitions (same as OpenAI pattern)
        self.tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Searches the user's long-term memory. REQUIRED: Use this tool whenever the user asks about themselves, past conversations, preferences, or established facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant memories.",
                            },
                            "search_tier": {
                                "type": "string",
                                "enum": ["fast", "balanced", "deep"],
                                "description": "The depth of search: 'fast' for simple lookups, 'balanced' for standard questions, 'deep' for complex reasoning.",
                            },
                        },
                        "required": ["query", "search_tier"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_task",
                    "description": "Schedules a task or reminder.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Description of the task.",
                            },
                            "due_date": {
                                "type": "string",
                                "description": "Due date in ISO 8601 format.",
                            },
                        },
                        "required": ["task_description", "due_date"],
                    },
                },
            },
        ]

    @property
    def curation_service(self) -> "CurationService":
        """Lazy-load the curation service."""
        if self._curation_service is None:
            from ..services import CurationService

            self._curation_service = CurationService(
                self.vector_storage,
                self.graph_storage,
                interval_seconds=self.curation_interval_seconds,
            )
            self._curation_service.start()
        return self._curation_service

    @property
    def embedding_model(self) -> "BaseEmbeddingModel":
        """Lazy-load the embedding model only when needed. Returns None in LIGHTWEIGHT mode."""
        if self.operation_mode == "lightweight":
            return None

        if self._embedding_model is None:
            if self._provided_embedding_model is None:
                # Use appropriate embedding model based on mode
                if self.operation_mode == "online":
                    from ..embedding_models import OpenAIEmbeddingModel

                    self._embedding_model = OpenAIEmbeddingModel(
                        client=self.client, model_name="text-embedding-3-small"
                    )
                else:  # local mode
                    from ..embedding_models import LocalEmbeddingModel

                    self._embedding_model = LocalEmbeddingModel()
            else:
                self._embedding_model = self._provided_embedding_model
        return self._embedding_model

    @property
    def vector_storage(self) -> "ChromaStorage":
        """Lazy-load vector storage only when needed. Returns None in LIGHTWEIGHT mode."""
        if self.operation_mode == "lightweight":
            return None

        if self._vector_storage is None:
            from ..storage.chroma import ChromaStorage

            self._vector_storage = ChromaStorage(
                self.storage_path, dimension=self.embedding_model.dimension
            )
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

            # Convert string mode to enum
            mode = SalienceMode(self.operation_mode.lower())

            # For LOCAL mode, share embedding model
            # For ONLINE mode, pass OpenAI API key
            self._salience_gate = SalienceGate(
                threshold=self.salience_threshold,
                embedding_model=self.embedding_model
                if mode == SalienceMode.LOCAL
                else None,
                mode=mode,
                openai_api_key=self.client.api_key
                if mode == SalienceMode.ONLINE
                else None,
            )
        return self._salience_gate

    @property
    def search_service(self) -> "SearchService":
        """Lazy-load search service only when needed."""
        if self._search_service is None:
            from ..services import SearchService

            self._search_service = SearchService(
                self.vector_storage, self.graph_storage, self.embedding_model
            )
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
                llm_client=self,
            )
        return self._consolidation_service

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        """
        Send a chat completion request with memory capabilities.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stream: If True, returns a generator that yields response chunks
            **kwargs: Additional arguments for the completion (will override defaults)

        Returns:
            str | Generator: The assistant's response (string if stream=False, generator if stream=True)
        """
        # Ensure curation service is started
        _ = self.curation_service

        self.last_trace = None  # Reset trace for each new chat call

        triggered_context = self.search_service.get_triggered_tasks_context(
            self.user_id
        )
        if triggered_context:
            # Prepend the task reminders as a system message to guide the LLM's response.
            messages.insert(0, {"role": "system", "content": triggered_context})

        user_query = (
            messages[-1]["content"]
            if messages and messages[-1]["role"] == "user"
            else ""
        )

        # Apply defaults if not overridden
        completion_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "tools": self.tool_schema,
            "tool_choice": "auto",
            "stream": stream,
        }
        completion_kwargs.update(kwargs)

        # Handle streaming mode
        if stream:
            return self._stream_chat(completion_kwargs, user_query)

        try:
            print(f"[Novita] Calling Novita API with model: {self.model}")
            chat_start = __import__("time").time()
            response = self.client.chat.completions.create(**completion_kwargs)
            chat_elapsed = __import__("time").time() - chat_start
            print(f"[Novita] Novita API call completed in {chat_elapsed:.2f}s")
            response_message = response.choices[0].message
        except Exception as e:
            print(f"Error during initial LLM call: {e}")
            return "Sorry, I encountered an error trying to process your request."

        # Check if the LLM decided to use our tool
        if not response_message.tool_calls:
            final_response = response_message.content
        else:
            # Handle multiple tool calls
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name

                if function_name == "search_memory":
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        search_tier = function_args.get("search_tier", "balanced")

                        search_output = self.search_service.search(
                            query=query,
                            user_id=self.user_id,
                            search_tier=search_tier,
                            llm_client=self,
                        )
                        search_result_text = search_output["result"]
                        self.last_trace = search_output["trace"]

                        # Append the tool result message
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": search_result_text,
                            }
                        )

                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError in search_memory tool call: {e}")
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": "Error: Failed to parse tool arguments.",
                            }
                        )
                    except Exception as e:
                        print(f"Error during search_memory tool execution: {e}")
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": "Sorry, I encountered an error while searching my memory.",
                            }
                        )
                        if self.last_trace:
                            self.last_trace.conclude(error=e)

                elif function_name == "schedule_task":
                    try:
                        import dateutil.parser

                        function_args = json.loads(tool_call.function.arguments)
                        description = function_args.get("task_description")
                        due_date_str = function_args.get("due_date")

                        due_timestamp = dateutil.parser.parse(due_date_str).timestamp()

                        task_id = self.graph_storage.add_task(
                            description, due_timestamp, self.user_id
                        )

                        tool_response = f"Task successfully scheduled with ID: {task_id}. I will remind you when it's due."

                    except ImportError:
                        print(
                            "Error: dateutil.parser is required for schedule_task. Install with: pip install python-dateutil"
                        )
                        tool_response = (
                            "Error: Missing required library for date parsing."
                        )
                    except Exception as e:
                        print(f"Error scheduling task: {e}")
                        tool_response = "Error: Could not schedule the task due to an invalid date format or other issue."

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_response,
                        }
                    )

                else:
                    print(f"Warning: LLM called unknown tool '{function_name}'")
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: Unknown tool '{function_name}'.",
                        }
                    )

            # After processing all tool calls, send results back to LLM
            try:
                clean_kwargs = {
                    k: v
                    for k, v in completion_kwargs.items()
                    if k not in ["tools", "tool_choice"]
                }
                clean_kwargs["messages"] = messages

                second_response = self.client.chat.completions.create(**clean_kwargs)
                final_response = second_response.choices[0].message.content
            except Exception as e:
                print(f"Error during second LLM call after tool execution: {e}")
                final_response = (
                    "Sorry, I encountered an error while processing the tool results."
                )

        # Consolidate the full interaction
        full_interaction = f"User: {user_query}\nAssistant: {final_response}"
        self.consolidation_service.consolidate(full_interaction, self.user_id)

        return final_response

    def _stream_chat(self, completion_kwargs: dict, user_query: str):
        """
        Helper method to handle streaming responses.

        Args:
            completion_kwargs: Arguments for the completion call
            user_query: The user's query for consolidation

        Yields:
            str: Response chunks from the LLM
        """
        try:
            stream_response = self.client.chat.completions.create(**completion_kwargs)

            full_response = ""
            tool_calls_buffer = []
            current_tool_call = None

            for chunk in stream_response:
                delta = chunk.choices[0].delta

                if delta.content:
                    full_response += delta.content
                    yield delta.content

                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        if tool_call_chunk.index is not None:
                            while len(tool_calls_buffer) <= tool_call_chunk.index:
                                tool_calls_buffer.append(
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )

                            current_tool_call = tool_calls_buffer[tool_call_chunk.index]

                            if tool_call_chunk.id:
                                current_tool_call["id"] = tool_call_chunk.id
                            if tool_call_chunk.type:
                                current_tool_call["type"] = tool_call_chunk.type
                            if tool_call_chunk.function:
                                if tool_call_chunk.function.name:
                                    current_tool_call["function"]["name"] = (
                                        tool_call_chunk.function.name
                                    )
                                if tool_call_chunk.function.arguments:
                                    current_tool_call["function"]["arguments"] += (
                                        tool_call_chunk.function.arguments
                                    )

            # Handle tool calls if any
            if tool_calls_buffer:
                messages = completion_kwargs["messages"].copy()
                messages.append(
                    {
                        "role": "assistant",
                        "content": full_response or None,
                        "tool_calls": tool_calls_buffer,
                    }
                )

                for tool_call in tool_calls_buffer:
                    function_name = tool_call["function"]["name"]

                    if function_name == "search_memory":
                        try:
                            function_args = json.loads(
                                tool_call["function"]["arguments"]
                            )
                            query = function_args.get("query")
                            search_tier = function_args.get("search_tier", "balanced")

                            search_output = self.search_service.search(
                                query=query,
                                user_id=self.user_id,
                                search_tier=search_tier,
                                llm_client=self,
                            )
                            search_result_text = search_output["result"]
                            self.last_trace = search_output["trace"]

                            messages.append(
                                {
                                    "tool_call_id": tool_call["id"],
                                    "role": "tool",
                                    "name": function_name,
                                    "content": search_result_text,
                                }
                            )
                        except Exception as e:
                            print(f"Error during search_memory: {e}")
                            messages.append(
                                {
                                    "tool_call_id": tool_call["id"],
                                    "role": "tool",
                                    "name": function_name,
                                    "content": "Error searching memory.",
                                }
                            )

                    elif function_name == "schedule_task":
                        try:
                            import dateutil.parser

                            function_args = json.loads(
                                tool_call["function"]["arguments"]
                            )
                            description = function_args.get("task_description")
                            due_date_str = function_args.get("due_date")
                            due_timestamp = dateutil.parser.parse(
                                due_date_str
                            ).timestamp()
                            task_id = self.graph_storage.add_task(
                                description, due_timestamp, self.user_id
                            )
                            tool_response = (
                                f"Task successfully scheduled with ID: {task_id}."
                            )
                        except Exception as e:
                            print(f"Error scheduling task: {e}")
                            tool_response = "Error: Could not schedule the task."

                        messages.append(
                            {
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": function_name,
                                "content": tool_response,
                            }
                        )

                clean_kwargs = {
                    k: v
                    for k, v in completion_kwargs.items()
                    if k not in ["tools", "tool_choice"]
                }
                clean_kwargs["messages"] = messages
                clean_kwargs["stream"] = True

                second_stream = self.client.chat.completions.create(**clean_kwargs)
                for second_chunk in second_stream:
                    if second_chunk.choices[0].delta.content:
                        yield second_chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error during streaming: {e}")
            yield "Sorry, I encountered an error while streaming the response."

        # Consolidate after streaming completes
        full_interaction = f"User: {user_query}\nAssistant: {full_response}"
        self.consolidation_service.consolidate(full_interaction, self.user_id)

    def analyze_and_extract_knowledge(self, text: str) -> Dict:
        """
        Extracts facts, entities, and relationships from text for the knowledge graph.
        Uses a fast, cheap model (deepseek/deepseek-v3.2) for efficiency.

        Args:
            text: The text to analyze

        Returns:
            Dict with keys 'facts', 'entities', and 'relationships'
        """
        import time
        from datetime import datetime

        start_time = time.time()
        print(
            f"[EXTRACTION] Starting knowledge extraction... (text length: {len(text)} chars)"
        )

        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p %Z")

        # Use a fast, cheap model for knowledge extraction instead of the main model
        extraction_model = (
            "deepseek/deepseek-chat"  # Fast and cheap for extraction tasks
        )

        system_prompt = f"""
You are a Knowledge Graph Engineer AI. Your task is to analyze text and deconstruct it into a structured knowledge graph.
The current date and time is {current_datetime}.
You must identify:
1.  **facts**: A list of simple, atomic statements. For each fact, assign an 'importance_score' (float 0.1-1.0) and an 'expiration_date' (ISO 8601 string or null if it doesn't expire).
2.  **entities**: A list of key nouns (people, places, projects). Each entity should have a 'name' and a 'type'.
3.  **relationships**: A list of connections between entities. Each relationship must have a 'subject' (entity name), a 'predicate' (the verb or connecting phrase), and an 'object' (entity name).

Respond ONLY with a valid JSON object.

Example Input:
"John confirmed the temporary door code is 1234 for the next 24 hours. This is for Project Phoenix, which is our top priority."

Example JSON Output:
{
            "facts": [
    {
                "fact": "The temporary door code is 1234.", "importance_score": 0.8, "expiration_date": "2025-11-16T14:30:00Z"},
    {
                "fact": "Project Phoenix is the team's top priority.", "importance_score": 1.0, "expiration_date": null}
  ],
  "entities": [
    {"name": "John", "type": "Person"},
    {"name": "Project Phoenix", "type": "Project"}
  ],
  "relationships": [
    {"subject": "John", "predicate": "works on", "object": "Project Phoenix"}
  ]
}
"""
        try:
            print(f"[EXTRACTION] Calling Novita API with model: {extraction_model}")
            api_start = time.time()

            response = self.client.chat.completions.create(
                model=extraction_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            api_elapsed = time.time() - api_start
            print(f"[EXTRACTION] Novita API call completed in {api_elapsed:.2f}s")

            content = response.choices[0].message.content
            if not content:
                print(
                    f"[EXTRACTION] No content in response. Total time: {time.time() - start_time:.2f}s"
                )
                return {"facts": [], "entities": [], "relationships": []}

            parse_start = time.time()
            knowledge_graph = json.loads(content)
            parse_elapsed = time.time() - parse_start
            print(f"[EXTRACTION] JSON parsing completed in {parse_elapsed:.2f}s")

            # Basic validation to ensure keys exist
            knowledge_graph.setdefault("facts", [])
            knowledge_graph.setdefault("entities", [])
            knowledge_graph.setdefault("relationships", [])
            for fact in knowledge_graph["facts"]:
                fact.setdefault("importance_score", 0.5)
                fact.setdefault("expiration_date", None)

            total_elapsed = time.time() - start_time
            print(
                f"[EXTRACTION] Knowledge extraction complete in {total_elapsed:.2f}s - Found {len(knowledge_graph.get('facts', []))} facts, {len(knowledge_graph.get('entities', []))} entities, {len(knowledge_graph.get('relationships', []))} relationships"
            )

            return knowledge_graph
        except Exception as e:
            print(
                f"[EXTRACTION] Error during knowledge extraction: {e} (after {time.time() - start_time:.2f}s)"
            )
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

        # Perform a guaranteed "deep" search
        search_output = self.search_service.search(
            query=question, user_id=self.user_id, search_tier="deep", llm_client=self
        )
        context = search_output["result"]
        self.last_trace = search_output["trace"]

        # Construct the Synthesis Prompt
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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.0,
            )
            answer_text = response.choices[0].message.content
        except Exception as e:
            print(f"Error during synthesis LLM call: {e}")
            answer_text = "Sorry, I encountered an error while synthesizing the answer."

        if return_object:
            from ..observability import AnswerObject

            return AnswerObject(
                question=question,
                answer=answer_text,
                context=context,
                trace=self.last_trace,
            )
        else:
            return answer_text

    def extract_query_entities(self, query: str) -> List[str]:
        """
        Uses a focused LLM call to identify key entities in a search query.
        """
        system_prompt = """
You are an efficient entity extraction model. Your task is to identify the main nouns or concepts in the user's query that could be looked up in a knowledge graph.
Do not explain. Do not use conversational filler. Respond ONLY with a valid JSON object containing a single key "entities", which is a list of the extracted entity strings.
If no specific entities are found, return an empty list.

Example 1:
Query: "Who is the lead engineer for Project Phoenix?)
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
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
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

    def close(self):
        """Release resources and close storage connections."""
        try:
            # Stop the curation service FIRST before closing storage
            if self._curation_service is not None:
                self._curation_service.stop()
            if self._vector_storage is not None:
                self._vector_storage.close()
            if self._graph_storage is not None:
                self._graph_storage.close()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
