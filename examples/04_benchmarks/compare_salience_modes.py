"""
Compare the three salience gate modes:
- LOCAL: Sentence-transformers (slow startup, high accuracy)
- ONLINE: OpenAI embeddings API (fast startup, API cost)
- LIGHTWEIGHT: Keyword-based (instant startup, lower accuracy)
"""
import os
import time
import shutil
from memlayer.wrappers.openai import OpenAI

def cleanup(path):
    """Remove test storage."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except:
            pass

def test_mode(mode: str, storage_path: str):
    """Test a specific salience mode."""
    cleanup(storage_path)
    
    print("\n" + "="*70)
    print(f"TESTING: {mode.upper()} MODE")
    print("="*70)
    
    # Time initialization
    print(f"\n1. Initializing client with {mode} mode...")
    init_start = time.time()
    client = OpenAI(
        storage_path=storage_path,
        user_id=f"test_user_{mode}",
        salience_mode=mode
    )
    init_time = time.time() - init_start
    print(f"   Client init: {init_time:.3f}s")
    
    # Time first chat (triggers lazy loading)
    print(f"\n2. First chat (triggers lazy loading)...")
    first_chat_start = time.time()
    response = client.chat(messages=[
        {"role": "user", "content": "Hello! My name is Alice and I work at TechCorp."}
    ])
    first_chat_time = time.time() - first_chat_start
    print(f"   First chat: {first_chat_time:.3f}s")
    print(f"   Response: {response[:60]}...")
    
    # Wait for consolidation
    time.sleep(2)
    
    # Second chat
    print(f"\n3. Second chat (no loading)...")
    second_chat_start = time.time()
    response = client.chat(messages=[
        {"role": "user", "content": "What company do I work for?"}
    ])
    second_chat_time = time.time() - second_chat_start
    print(f"   Second chat: {second_chat_time:.3f}s")
    print(f"   Response: {response[:60]}...")
    
    # Wait for final consolidation to complete
    time.sleep(3)
    
    # Cleanup
    client.close()
    time.sleep(1)  # Give more time for cleanup
    cleanup(storage_path)
    
    return {
        "mode": mode,
        "init_time": init_time,
        "first_chat_time": first_chat_time,
        "second_chat_time": second_chat_time,
        "total_first_use": init_time + first_chat_time
    }

def main():
    """Compare all three salience modes."""
    print("="*70)
    print("SALIENCE GATE MODE COMPARISON")
    print("="*70)
    print("\nThis benchmark compares three salience gate modes:")
    print("1. LOCAL: Uses sentence-transformers (slow startup, no API cost)")
    print("2. ONLINE: Uses OpenAI embeddings API (fast startup, small API cost)")
    print("3. LIGHTWEIGHT: Uses keywords only (instant startup, no cost)")
    print()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable.")
        return
    
    results = []
    
    # Test all three modes
    for mode in ["lightweight", "online", "local"]:
        try:
            result = test_mode(mode, f"./test_salience_{mode}")
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR in {mode} mode: {e}")
            continue
        
        # Clear cache between modes
        from memlayer.embedding_models import _MODEL_CACHE
        _MODEL_CACHE.clear()
        time.sleep(1)
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print()
    
    print(f"{'Mode':<15} {'Init Time':<15} {'First Chat':<15} {'Total First Use':<15}")
    print("-" * 70)
    
    for result in results:
        print(
            f"{result['mode'].upper():<15} "
            f"{result['init_time']:>13.3f}s "
            f"{result['first_chat_time']:>13.3f}s "
            f"{result['total_first_use']:>13.3f}s"
        )
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if len(results) >= 3:
        lightweight = next(r for r in results if r['mode'] == 'lightweight')
        online = next(r for r in results if r['mode'] == 'online')
        local = next(r for r in results if r['mode'] == 'local')
        
        print("\n[LIGHTWEIGHT Mode]:")
        print(f"   - Fastest startup: {lightweight['total_first_use']:.1f}s")
        print(f"   - No API costs, no model downloads, no embeddings")
        print(f"   - Graph-only storage (no vector search)")
        print(f"   - Best for: Prototyping, low-resource environments")
        print(f"   - Trade-off: No semantic search, keyword-based only")
        
        print("\n[ONLINE Mode]:")
        print(f"   - Moderate startup: {online['total_first_use']:.1f}s")
        print(f"   - Saves ~{local['total_first_use'] - online['total_first_use']:.1f}s vs LOCAL")
        print(f"   - API cost: ~$0.0002 per operation")
        print(f"   - Best for: Production apps, serverless functions")
        print(f"   - Trade-off: Requires internet, API cost")
        
        print("\n[LOCAL Mode]:")
        print(f"   - Slower startup: {local['total_first_use']:.1f}s")
        print(f"   - High accuracy with semantic understanding")
        print(f"   - No ongoing costs, works offline")
        print(f"   - Best for: High-volume usage, offline apps")
        print(f"   - Trade-off: Slow startup (model loading)")
        
        print("\n[RECOMMENDATION]:")
        if lightweight['total_first_use'] < 5:
            print("   Use LIGHTWEIGHT for quick prototyping and testing!")
        if online['total_first_use'] < local['total_first_use'] / 2:
            print("   Use ONLINE for production apps with fast cold starts!")
        print("   Use LOCAL for high-volume apps where startup time doesn't matter!")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
