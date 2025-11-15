from .storage.chroma import ChromaStorage
from .storage.memgraph import MemgraphStorage
from .wrappers.openai import OpenAIWrapper
from .wrappers.ollama import OllamaWrapper
from .wrappers.gemini import GeminiWrapper
from .wrappers.claude import ClaudeWrapper
from .embedding_models import BaseEmbeddingModel, OpenAIEmbeddingModel, LocalEmbeddingModel
import openai
from .ml_gate import SalienceGate
from .storage.networkx import NetworkXStorage
from .services import SchedulerService # <-- Import the new service
import atexit
class Memory:
    """
    The main client for interacting with the Memory Bank.
    """
    def __init__(self, 
                 mode: str = "embedded", 
                 storage_path: str = "./memory_bank_data", 
                 user_id: str = "default_user",
                 embedding_model: BaseEmbeddingModel = None,
                 salience_threshold: float = 0.1,
                 scheduler_interval_seconds: int = 60):

        self.user_id = user_id
        
        # --- Embedding Model Configuration ---
        # Set this FIRST before using it
        if embedding_model is None:
            # Default to OpenAI if not specified
            print("No embedding model provided. Defaulting to OpenAI 'text-embedding-3-small'.")
            self.embedding_model = OpenAIEmbeddingModel(client=openai.OpenAI())
        else:
            self.embedding_model = embedding_model
        # ------------------------------------
        
        self.salience_gate = SalienceGate(threshold=salience_threshold)

        if mode == "embedded":
            print(f"Initializing Memory Bank in embedded mode at '{storage_path}'...")
            # Pass the embedding model's dimension to the storage
            self.vector_storage = ChromaStorage(storage_path, dimension=self.embedding_model.dimension)
            self.graph_storage = NetworkXStorage(storage_path)
            print("Initializing SchedulerService...")
            self.scheduler = SchedulerService(
                graph_storage=self.graph_storage,
                check_interval_seconds=scheduler_interval_seconds
            )
            self.scheduler.start()
            
            # Register the close method to be called upon script exit
            atexit.register(self.close)
        elif mode in ["self-hosted", "cloud"]:
            raise NotImplementedError("Self-hosted and cloud modes are not yet available.")
        else:
            raise ValueError("Invalid mode specified. Only 'embedded' is currently supported.")
    def close(self):
        """
        Gracefully shuts down the Memory Bank's background services.
        This is registered with atexit to be called automatically.
        """
        print("Closing Memory Bank...")
        if hasattr(self, 'scheduler') and self.scheduler:
            self.scheduler.stop()
    def wrap(self, llm_client):
        client_type_str = str(type(llm_client)).lower()
        
        wrapper_args = {
            "vector_storage": self.vector_storage,
            "graph_storage": self.graph_storage,
            "embedding_model": self.embedding_model,
            "salience_gate": self.salience_gate, # <-- Pass the gate instance

            "user_id": self.user_id
        }

        if "openai" in client_type_str:
            print("OpenAI client detected. Applying memory wrapper.")
            return OpenAIWrapper(client=llm_client, **wrapper_args)
        
        elif isinstance(llm_client, dict) and "ollama" in llm_client.get("provider", ""):
            print("Ollama client configuration detected. Applying memory wrapper.")
            return OllamaWrapper(client_config=llm_client, **wrapper_args)
            
        elif "google" in client_type_str and ("genai" in client_type_str or "client" in client_type_str):
            print("Google Gemini client detected. Applying memory wrapper.")
            return GeminiWrapper(client=llm_client, **wrapper_args)
            
        elif "anthropic" in client_type_str:
            print("Anthropic Claude client detected. Applying memory wrapper.")
            return ClaudeWrapper(client=llm_client, **wrapper_args)
            
        else:
            raise TypeError(f"Unsupported LLM client type: {type(llm_client)}. Supported: OpenAI, Ollama, Gemini, Claude.")