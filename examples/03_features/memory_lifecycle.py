import time
import os
import shutil
from datetime import datetime, timedelta
from memlayer.wrappers.openai import OpenAI

# --- Configuration ---
STORAGE_PATH = "./lifecycle_test_memory"
USER_ID = "lifecycle_user_001"

# --- Helper Functions ---
def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def cleanup():
    """Removes the storage directory created during the test."""
    if os.path.exists(STORAGE_PATH):
        try:
            shutil.rmtree(STORAGE_PATH)
        except PermissionError as e:
            # On Windows, ChromaDB may still have file locks even after close()
            # Try again after a short delay
            print(f"Warning: {e}")
            print("Retrying cleanup after 1 second...")
            time.sleep(1)
            try:
                shutil.rmtree(STORAGE_PATH)
            except PermissionError:
                print(f"Warning: Could not delete {STORAGE_PATH} due to file locks.")
                print("Please delete it manually or restart your terminal.")

def get_memory_state(client: OpenAI, memory_id: str) -> dict:
    """A helper to inspect the state of a specific memory node in the graph."""
    if client.graph_storage.graph.has_node(memory_id):
        return client.graph_storage.graph.nodes[memory_id]
    return {}

# --- Main Test Execution ---
def run_example():
    """
    This example demonstrates the full memory lifecycle:
    1.  Create memories with different importance scores and one with an expiration date.
    2.  Access one memory repeatedly to track "attention".
    3.  Simulate the passage of time.
    4.  Trigger the curation service to observe decay (archival) and expiration (deletion).
    """
    # Ensure the OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable to run this example.")
        return

    cleanup()

    print_header("Step 1: Initialize Client & Create Memories")
    
    # We use a very short curation interval for the demo to see results quickly.
    client = OpenAI(
        storage_path=STORAGE_PATH,
        user_id=USER_ID,
        curation_interval_seconds=10 # Run curation every 10 seconds
    )
    
    # We need to manually add memories for this test to control their properties.
    # We'll create a few nodes directly in the graph for demonstration.
    graph = client.graph_storage
    
    # Memory 1: A critical, permanent fact
    CRITICAL_FACT_ID = "Project Phoenix"
    graph.add_entity(CRITICAL_FACT_ID, "Project", metadata={"importance_score": 1.0})
    
    # Memory 2: A trivial, unimportant fact
    TRIVIAL_FACT_ID = "User likes coffee"
    graph.add_entity(TRIVIAL_FACT_ID, "Preference", metadata={"importance_score": 0.2})
    
    # Memory 3: A temporary, expiring fact
    EXPIRING_FACT_ID = "Temporary Access Code"
    expiration_time = time.time() + 25 # Expires in 25 seconds
    graph.add_entity(EXPIRING_FACT_ID, "Credential", metadata={
        "importance_score": 0.9,
        "expiration_timestamp": expiration_time
    })
    
    print(f"Created 3 memories for user '{USER_ID}':")
    print(f"  - CRITICAL: '{CRITICAL_FACT_ID}' (Importance: 1.0)")
    print(f"  - TRIVIAL: '{TRIVIAL_FACT_ID}' (Importance: 0.2)")
    print(f"  - EXPIRING: '{EXPIRING_FACT_ID}' (Expires in 25s)")

    # Force the curation service to start by accessing the property
    _ = client.curation_service
    print(f"Curation service started with interval: {client.curation_interval_seconds}s")

    # --------------------------------------------------------------------------
    
    print_header("Step 2: Track Attention by Accessing a Memory")
    print("Simulating 5 accesses to the CRITICAL fact...")
    
    # We'll call the search service's attention tracker directly for this test.
    # In a real scenario, this would happen during a `client.chat()` call.
    client.search_service._track_attention(vector_results=[], graph_results=[f"(Project) {CRITICAL_FACT_ID} --[related_to]--> (Something) Else"])
    client.search_service._track_attention(vector_results=[], graph_results=[f"(Project) {CRITICAL_FACT_ID} --[related_to]--> (Something) Else"])
    client.search_service._track_attention(vector_results=[], graph_results=[f"(Project) {CRITICAL_FACT_ID} --[related_to]--> (Something) Else"])
    client.search_service._track_attention(vector_results=[], graph_results=[f"(Project) {CRITICAL_FACT_ID} --[related_to]--> (Something) Else"])
    client.search_service._track_attention(vector_results=[], graph_results=[f"(Project) {CRITICAL_FACT_ID} --[related_to]--> (Something) Else"])
    
    critical_state = get_memory_state(client, CRITICAL_FACT_ID)
    print(f"  - State of '{CRITICAL_FACT_ID}': Access Count = {critical_state.get('access_count')}")
    assert critical_state.get('access_count') == 5

    # --------------------------------------------------------------------------

    print_header("Step 3: Simulate Time Passing & Trigger Curation")
    wait_time = 32  # Wait a bit longer than 30s to ensure the 4th cycle runs
    print(f"Waiting for {wait_time} seconds. The CurationService will run in the background.")
    print("During this time:")
    print("  - The TRIVIAL fact should decay and be ARCHIVED.")
    print("  - The EXPIRING fact should be permanently DELETED.")
    print("  - The CRITICAL fact should remain ACTIVE due to high importance and recent access.")
    
    time.sleep(wait_time)

    # --------------------------------------------------------------------------

    print_header("Step 4: Verify Final Memory States")
    
    # Reload from disk to ensure persistence of curation changes
    client.close() # Stop the first client's background threads
    reloaded_client = OpenAI(storage_path=STORAGE_PATH, user_id=USER_ID)

    # Check Critical Fact
    critical_state = get_memory_state(reloaded_client, CRITICAL_FACT_ID)
    print(f"\nVerifying '{CRITICAL_FACT_ID}'...")
    if critical_state.get('status') == 'active':
        print(f"  [PASS] Status is 'active' as expected.")
    else:
        print(f"  [FAIL] Status is '{critical_state.get('status')}', but should be 'active'.")

    # Check Trivial Fact
    trivial_state = get_memory_state(reloaded_client, TRIVIAL_FACT_ID)
    print(f"\nVerifying '{TRIVIAL_FACT_ID}'...")
    if trivial_state.get('status') == 'archived':
        print(f"  [PASS] Status is 'archived' as expected due to decay.")
    else:
        print(f"  [FAIL] Status is '{trivial_state.get('status')}', but should be 'archived'.")

    # Check Expiring Fact
    expiring_state = get_memory_state(reloaded_client, EXPIRING_FACT_ID)
    print(f"\nVerifying '{EXPIRING_FACT_ID}'...")
    if not expiring_state:
        print(f"  [PASS] Memory has been permanently deleted as expected due to expiration.")
    else:
        print(f"  [FAIL] Memory still exists, but should have been deleted.")

    # --------------------------------------------------------------------------

    reloaded_client.close()
    
    # Give Windows more time to release file handles after close()
    # ChromaDB on Windows can be slow to release SQLite database locks
    print("\nWaiting for file handles to be released...")
    import gc
    gc.collect()  # Force garbage collection
    time.sleep(2)  # Wait 2 seconds
    
    cleanup()
    print_header("Example Complete")

if __name__ == "__main__":
    run_example()