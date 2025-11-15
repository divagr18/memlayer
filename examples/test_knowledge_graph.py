import time
import shutil
import openai
from memory_bank import Memory
from memory_bank.embedding_models import LocalEmbeddingModel

# --- Test Configuration ---
TEST_USER_ID = "test_user_kg_nx_001"
STORAGE_PATH = "./test_kg_memory_nx" # Use a new path to avoid conflicts
TEST_CONVERSATION = (
    "User: Hey, just wanted to confirm I'm flying to Tokyo next Tuesday for the "
    "Project Phoenix meeting. My flight number is AC005. Also, I really prefer "
    "aisle seats if you can note that down. "
    "Assistant: Absolutely. I've noted your preference for aisle seats and the "
    "details for your trip to Tokyo for the Project Phoenix meeting."
)

# --- Helper Functions for Verification ---

def verify_vector_storage(memory_instance):
    """Displays the facts stored in ChromaDB."""
    print("\n--- Vector Storage (ChromaDB) Contents ---")
    try:
        results = memory_instance.vector_storage.collection.get(
            where={"user_id": TEST_USER_ID},
            include=["metadatas"]
        )
        
        stored_contents = [meta['content'] for meta in results.get('metadatas', [])]
        
        print(f"\nExtracted Facts ({len(stored_contents)} total):")
        for i, fact in enumerate(stored_contents, 1):
            print(f"  {i}. {fact}")
            
        return len(stored_contents) > 0
    except Exception as e:
        print(f"  [FAIL] An error occurred during vector verification: {e}")
        return False

def verify_graph_storage(memory_instance):
    """Displays the extracted entities and relationships from the NetworkX graph."""
    print("\n--- Graph Storage (NetworkX) Contents ---")
    try:
        graph = memory_instance.graph_storage.graph
        
        # Display Entities
        print(f"\nExtracted Entities ({graph.number_of_nodes()} total):")
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            print(f"  • {node} ({node_type})")

        # Display Relationships
        print(f"\nExtracted Relationships ({graph.number_of_edges()} total):")
        for subject, obj, data in graph.edges(data=True):
            predicate = data.get('type', 'related to')
            print(f"  • {subject} --[{predicate}]--> {obj}")
                
        return graph.number_of_nodes() > 0 and graph.number_of_edges() > 0
    except Exception as e:
        print(f"  [FAIL] An error occurred during graph verification: {e}")
        return False

# --- Main Test Execution ---
def run_test():
    print("Starting Knowledge Graph Consolidation Test with NetworkX...")
    
    # 1. Setup
    try:
        openai_client = openai.OpenAI()
    except openai.OpenAIError:
        print("ERROR: Please set your OPENAI_API_KEY environment variable to run this test.")
        return

    memory = Memory(
        storage_path=STORAGE_PATH,
        user_id=TEST_USER_ID,
        embedding_model=LocalEmbeddingModel()
    )
    
    wrapper = memory.wrap(openai_client)
    consolidation_service = wrapper.consolidation_service

    # 2. Action: Trigger the consolidation process
    print(f"\nConsolidating test conversation for user '{TEST_USER_ID}'...")
    consolidation_service.consolidate(TEST_CONVERSATION, TEST_USER_ID)

    # 3. Verification
    print("\nWaiting for consolidation to complete (15 seconds)...")
    time.sleep(15)

    # Reload the memory instance from disk to ensure persistence is working
    print("\nReloading memory from disk to verify persistence...")
    reloaded_memory = Memory(
        storage_path=STORAGE_PATH,
        user_id=TEST_USER_ID,
        embedding_model=LocalEmbeddingModel()
    )

    vector_result = verify_vector_storage(reloaded_memory)
    graph_result = verify_graph_storage(reloaded_memory)

    print("\n--- Test Summary ---")
    if vector_result and graph_result:
        print("✅  SUCCESS: NetworkX Knowledge Graph pipeline test passed!")
    else:
        print("❌  FAILURE: NetworkX Knowledge Graph pipeline test failed.")
        print(f"Please check the logs above and inspect the storage at '{STORAGE_PATH}'.")
        print(f"Specifically, check the file: {reloaded_memory.graph_storage.graph_path}")

if __name__ == "__main__":
    # Clean up previous test runs before starting
    shutil.rmtree(STORAGE_PATH, ignore_errors=True)
    run_test()