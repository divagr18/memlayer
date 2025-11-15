import chromadb
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

class ChromaStorage:
    """
    A vector storage backend using the embedded, on-disk version of ChromaDB.
    """
    def __init__(self, storage_path: str, dimension: int): # <-- Accept dimension
        self.db_path = str(Path(storage_path) / "chroma")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # We need to create a custom embedding function that does nothing,
        # since we will be providing the embeddings directly.
        class NoOpEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                # This should not be called if we always provide embeddings.
                # The dimension is what's important for collection creation.
                return []

        self.collection = self.client.get_or_create_collection(
            name=f"memories_dim_{dimension}", # <-- Collection name includes dimension
            embedding_function=NoOpEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"} # ChromaDB infers dimension from embeddings
        )
        print(f"Memory Bank (ChromaDB) initialized at: {self.db_path} for dimension {dimension}")

    def add_memory(self, content: str, embedding: List[float], user_id: str = "default_user", metadata: Dict = None):
        """Adds a new memory to the Chroma collection."""
        doc_metadata = metadata or {}
        doc_metadata.update({
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "content": content
        })
        
        self.collection.add(
            embeddings=[embedding],
            metadatas=[doc_metadata],
            documents=[content[:100]], # Document is a snippet for Chroma's internal use
            ids=[f"mem_{datetime.now(timezone.utc).timestamp()}"] # Simple unique ID
        )

    def search_memories(self, query_embedding: List[float], user_id: str = "default_user", top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches for the most similar memories for a given user."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id}
        )
        
        memories = []
        if not results or not results.get('metadatas') or not results.get('distances'):
            return []

        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        for i, meta in enumerate(metadatas):
            memories.append({
                "id": ids[i],
                "content": meta.get("content", ""),
                "timestamp": datetime.fromtimestamp(meta.get("timestamp", 0), tz=timezone.utc),
                "metadata": meta,
                "score": 1 - distances[i]
            })
            
        return memories