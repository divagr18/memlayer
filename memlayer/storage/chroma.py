import time
import uuid
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
        print(f"MemLayer (ChromaDB) initialized at: {self.db_path} for dimension {dimension}")

    def add_memory(self, content: str, embedding: List[float], user_id: str = "default_user", metadata: Dict = None):
        """Adds a new memory with initial lifecycle metadata."""
        doc_metadata = metadata or {}
        # --- NEW: Add lifecycle attributes at creation ---
        base_attrs = {
            "user_id": user_id,
            "timestamp": time.time(),
            "content": content,
            "status": "active",
            "access_count": 0,
            "last_accessed_timestamp": time.time(),
            "importance_score": 0.5,
        }
        base_attrs.update(doc_metadata)
        
        # ChromaDB only accepts str, int, float, or bool - filter out None values
        base_attrs = {k: v for k, v in base_attrs.items() if v is not None}
        
        memory_id = f"mem_{uuid.uuid4().hex}"
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[base_attrs],
            documents=[content[:100]]
        )

    def search_memories(self, query_embedding: List[float], user_id: str = "default_user", top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches only for 'active' memories."""
        # --- NEW: Filter out archived memories ---
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"status": {"$eq": "active"}}
            ]
        }
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
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

    def close(self):
        """Close the ChromaDB client and release file locks."""
        try:
            # Clear collection reference first
            self.collection = None
            
            # Clear client reference
            if self.client is not None:
                # ChromaDB doesn't have an explicit close, but clearing the reference
                # and forcing garbage collection helps on Windows
                self.client = None
                
                # Force garbage collection to help release file handles
                import gc
                gc.collect()
        except Exception as e:
            print(f"Warning: Error during ChromaDB cleanup: {e}")
    def track_memory_access(self, memory_ids: List[str]):
        """Increments access count and updates timestamp for given memory IDs."""
        if not memory_ids: return
        
        current_metadatas = self.collection.get(ids=memory_ids, include=["metadatas"])['metadatas']
        new_metadatas = []
        for meta in current_metadatas:
            meta['access_count'] = meta.get('access_count', 0) + 1
            meta['last_accessed_timestamp'] = time.time()
            new_metadatas.append(meta)
        
        if new_metadatas:
            self.collection.update(ids=memory_ids, metadatas=new_metadatas)

    def get_all_memories_for_curation(self) -> List[Dict]:
        """Returns all memories with their lifecycle metadata."""
        # Note: Chroma's get() without IDs can be slow on huge collections.
        # For production, this might need batching. For now, this is fine.
        results = self.collection.get(include=["metadatas"])
        memories = []
        for i, meta in enumerate(results['metadatas']):
            memory_data = meta.copy()
            memory_data['id'] = results['ids'][i]
            memories.append(memory_data)
        return memories

    def update_memory_status(self, memory_id: str, new_status: str):
        """Updates the status of a memory (e.g., to 'archived')."""
        try:
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            if result['metadatas'] and len(result['metadatas']) > 0:
                current_meta = result['metadatas'][0]
                current_meta['status'] = new_status
                self.collection.update(ids=[memory_id], metadatas=[current_meta])
        except Exception as e:
            # Memory might not exist in vector store (e.g., only in graph)
            pass

    def delete_memory(self, memory_id: str):
        """Permanently deletes a memory."""
        try:
            self.collection.delete(ids=[memory_id])
        except Exception as e:
            # Memory might not exist in vector store
            pass