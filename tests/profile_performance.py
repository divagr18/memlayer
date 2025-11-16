"""
Performance profiling script for the MemLayer system.
This script profiles key operations to identify bottlenecks.
"""
import cProfile
import pstats
import os
import shutil
from io import StringIO
from memlayer.wrappers.openai import OpenAI

# Configuration
STORAGE_PATH = "./profile_test_memory"
USER_ID = "profile_user"

def cleanup():
    """Remove test storage."""
    if os.path.exists(STORAGE_PATH):
        try:
            shutil.rmtree(STORAGE_PATH)
        except:
            pass

def profile_initialization():
    """Profile client initialization time."""
    print("\n" + "="*70)
    print("PROFILING: Client Initialization")
    print("="*70)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    client = OpenAI(
        storage_path=STORAGE_PATH,
        user_id=USER_ID
    )
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    return client

def profile_first_chat(client):
    """Profile first chat interaction (includes lazy loading)."""
    print("\n" + "="*70)
    print("PROFILING: First Chat (with lazy loading)")
    print("="*70)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    response = client.chat(messages=[
        {"role": "user", "content": "Hello! My name is Alice and I love hiking in the mountains."}
    ])
    
    profiler.disable()
    
    print(f"Response: {response[:100]}...")
    
    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions
    print(s.getvalue())

def profile_memory_search(client):
    """Profile memory search operation."""
    print("\n" + "="*70)
    print("PROFILING: Memory Search (fast tier)")
    print("="*70)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    response = client.chat(messages=[
        {"role": "user", "content": "What do you know about my hobbies?"}
    ])
    
    profiler.disable()
    
    print(f"Response: {response[:100]}...")
    
    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    print(s.getvalue())

def profile_deep_search(client):
    """Profile deep tier search with graph traversal."""
    print("\n" + "="*70)
    print("PROFILING: Deep Search (with graph traversal)")
    print("="*70)
    
    # Add more context first
    client.chat(messages=[
        {"role": "user", "content": "I work as a software engineer at TechCorp in Seattle."}
    ])
    
    client.chat(messages=[
        {"role": "user", "content": "My best friend is Bob, we went to college together."}
    ])
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    response = client.chat(messages=[
        {"role": "user", "content": "Tell me everything you know about me."}
    ])
    
    profiler.disable()
    
    print(f"Response: {response[:100]}...")
    
    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    print(s.getvalue())

def main():
    """Run all profiling tests."""
    cleanup()
    
    print("="*70)
    print("MemLayer PERFORMANCE PROFILING")
    print("="*70)
    print("This will profile key operations to identify bottlenecks.")
    print()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable.")
        return
    
    # Profile initialization
    client = profile_initialization()
    
    # Profile first chat (includes lazy loading of embedding model, etc.)
    profile_first_chat(client)
    
    # Profile memory search
    profile_memory_search(client)
    
    # Profile deep search
    profile_deep_search(client)
    
    # Cleanup
    print("\n" + "="*70)
    print("Cleaning up...")
    client.close()
    cleanup()
    
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print("\nKey areas to examine:")
    print("1. Embedding model initialization time")
    print("2. Vector search performance")
    print("3. Graph traversal overhead")
    print("4. Salience gate evaluation")
    print("5. Knowledge extraction/consolidation")

if __name__ == "__main__":
    main()
