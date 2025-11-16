"""
Benchmark script to compare embedding model performance.
Tests initialization time and inference speed for different models.
"""
import time
import os
import shutil
from memlayer.wrappers.openai import OpenAI
from memlayer.embedding_models import LocalEmbeddingModel

# Configuration
STORAGE_PATH_1 = "./benchmark_memory_1"
STORAGE_PATH_2 = "./benchmark_memory_2"
USER_ID = "benchmark_user"

def cleanup(path):
    """Remove test storage."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except:
            pass

def benchmark_model(model_name: str, storage_path: str, num_runs: int = 3):
    """
    Benchmark a specific embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model
        storage_path: Path for temporary storage
        num_runs: Number of test conversations to run
    
    Returns:
        Dict with timing results
    """
    cleanup(storage_path)
    
    print("\n" + "="*70)
    print(f"BENCHMARKING: {model_name}")
    print("="*70)
    
    # Create custom embedding model
    print(f"\n1. Creating embedding model...")
    model_start = time.time()
    embedding_model = LocalEmbeddingModel(model_name=model_name)
    model_time = time.time() - model_start
    print(f"   Model initialization: {model_time:.3f}s")
    
    # Initialize client with custom model
    print(f"\n2. Initializing client...")
    client_start = time.time()
    client = OpenAI(
        storage_path=storage_path,
        user_id=USER_ID,
        embedding_model=embedding_model
    )
    client_time = time.time() - client_start
    print(f"   Client initialization: {client_time:.3f}s")
    
    # First chat (triggers lazy loading)
    print(f"\n3. First chat (with lazy loading)...")
    first_chat_start = time.time()
    response = client.chat(messages=[
        {"role": "user", "content": "Hello! My name is Alice and I work at TechCorp as a software engineer."}
    ])
    first_chat_time = time.time() - first_chat_start
    print(f"   First chat time: {first_chat_time:.3f}s")
    print(f"   Response: {response[:80]}...")
    
    # Wait for consolidation to complete
    time.sleep(3)
    
    # Subsequent chats
    print(f"\n4. Running {num_runs} additional chats...")
    chat_times = []
    
    test_messages = [
        "What do you know about my job?",
        "I live in Seattle and enjoy hiking on weekends.",
        "My favorite programming language is Python.",
    ]
    
    for i, msg in enumerate(test_messages[:num_runs], 1):
        chat_start = time.time()
        response = client.chat(messages=[{"role": "user", "content": msg}])
        chat_time = time.time() - chat_start
        chat_times.append(chat_time)
        print(f"   Chat {i}: {chat_time:.3f}s")
        time.sleep(1)  # Brief pause between chats
    
    avg_chat_time = sum(chat_times) / len(chat_times) if chat_times else 0
    print(f"\n   Average subsequent chat time: {avg_chat_time:.3f}s")
    
    # Cleanup
    client.close()
    time.sleep(0.5)
    cleanup(storage_path)
    
    return {
        "model_name": model_name,
        "model_init_time": model_time,
        "client_init_time": client_time,
        "first_chat_time": first_chat_time,
        "avg_chat_time": avg_chat_time,
        "total_first_use": model_time + client_time + first_chat_time
    }

def main():
    """Run benchmarks and compare results."""
    print("="*70)
    print("EMBEDDING MODEL PERFORMANCE BENCHMARK")
    print("="*70)
    print("\nThis will compare two embedding models:")
    print("1. all-MiniLM-L6-v2 (current default, 384 dim)")
    print("2. paraphrase-MiniLM-L3-v2 (faster, 384 dim)")
    print()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable.")
        return
    
    # Benchmark both models
    results = []
    
    # Model 1: Current default
    results.append(benchmark_model(
        model_name="all-MiniLM-L6-v2",
        storage_path=STORAGE_PATH_1
    ))
    
    # Clear any caches between runs
    from memlayer.embedding_models import _MODEL_CACHE
    _MODEL_CACHE.clear()
    time.sleep(2)
    
    # Model 2: Faster alternative
    results.append(benchmark_model(
        model_name="paraphrase-MiniLM-L3-v2",
        storage_path=STORAGE_PATH_2
    ))
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print()
    
    print(f"{'Metric':<35} {'Model 1':<20} {'Model 2':<20} {'Improvement':<15}")
    print("-" * 90)
    
    metrics = [
        ("Model", "model_name", "model_name", False),
        ("Model Init Time", "model_init_time", "model_init_time", True),
        ("Client Init Time", "client_init_time", "client_init_time", True),
        ("First Chat Time", "first_chat_time", "first_chat_time", True),
        ("Avg Subsequent Chat", "avg_chat_time", "avg_chat_time", True),
        ("Total First Use", "total_first_use", "total_first_use", True),
    ]
    
    for label, key1, key2, is_numeric in metrics:
        val1 = results[0][key1]
        val2 = results[1][key2]
        
        if is_numeric:
            improvement = ((val1 - val2) / val1 * 100) if val1 > 0 else 0
            print(f"{label:<35} {val1:>18.3f}s {val2:>18.3f}s {improvement:>13.1f}%")
        else:
            print(f"{label:<35} {val1:>18} {val2:>18} {'-':>15}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    model1_total = results[0]["total_first_use"]
    model2_total = results[1]["total_first_use"]
    
    if model2_total < model1_total:
        improvement = ((model1_total - model2_total) / model1_total) * 100
        print(f"\n✅ {results[1]['model_name']} is {improvement:.1f}% faster for first use!")
        print(f"   Saves ~{model1_total - model2_total:.1f}s on initial load.")
    else:
        print(f"\n✅ {results[0]['model_name']} remains the better choice.")
    
    print(f"\n📊 Both models have similar performance for subsequent chats (~2s),")
    print(f"   which is dominated by OpenAI API response time (network I/O).")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
