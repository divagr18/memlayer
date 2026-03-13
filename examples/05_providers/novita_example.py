"""
Novita.ai Provider Example

Demonstrates using Memlayer with Novita's OpenAI-compatible API.
Novita provides access to multiple open-source models like DeepSeek, GLM, and more.

Supported models:
- deepseek/deepseek-v3.2 (default)
- zai-org/glm-5
- minimax/minimax-m2.5
"""

from memlayer.wrappers.novita import Novita
import os

# Check for API key
if not os.getenv("NOVITA_API_KEY"):
    print("ERROR: Please set your NOVITA_API_KEY environment variable.")
    print("export NOVITA_API_KEY='your-api-key-here'")
    print("Get your API key from: https://novita.ai/")
    exit(1)

print("=" * 70)
print("Memlayer - NOVITA.AI EXAMPLE")
print("=" * 70)

# Initialize the memory-enhanced Novita client
client = Novita(
    # API key (optional - reads from NOVITA_API_KEY env var by default)
    # api_key="your-api-key-here",
    # Model selection
    # Default: deepseek/deepseek-v3.2
    # Other options: zai-org/glm-5, minimax/minimax-m2.5
    model="deepseek/deepseek-v3.2",
    # Standard OpenAI-compatible parameters
    temperature=0.7,
    # Memlayer settings
    storage_path="./novita_memories",
    user_id="demo_user",
    # Operation mode: "local" (default), "online", or "lightweight"
    # - local: Uses sentence-transformers (high accuracy, slow startup)
    # - online: Uses OpenAI embeddings API (fast startup, API cost)
    # - lightweight: Keyword-based only (instant startup, no embeddings)
    operation_mode="local",
    # Salience threshold: how strict to be about saving memories
    # -0.1 = permissive, 0.0 = balanced (default), 0.1 = strict
)

print("\n📝 Conversation 1: Teaching the AI about yourself")
print("-" * 70)

response = client.chat(
    messages=[
        {
            "role": "user",
            "content": "Hi! My name is Alex and I'm a machine learning engineer at DataCorp.",
        }
    ]
)
print(f"Assistant: {response}")

response = client.chat(
    messages=[
        {
            "role": "user",
            "content": "I'm working on Project Alpha, which uses Python and PyTorch.",
        }
    ]
)
print(f"\nAssistant: {response}")

response = client.chat(
    messages=[
        {
            "role": "user",
            "content": "My favorite programming language is Python and I prefer CPUs over GPUs for local work.",
        }
    ]
)
print(f"\nAssistant: {response}")

# Wait for background consolidation
print("\n⏳ Waiting for memory consolidation...")
import time

time.sleep(3)

print("\n🔍 Conversation 2: Testing memory recall")
print("-" * 70)

response = client.chat(
    messages=[{"role": "user", "content": "What do you know about me?"}]
)
print(f"Assistant: {response}")

response = client.chat(
    messages=[
        {
            "role": "user",
            "content": "What project am I working on and what technologies does it use?",
        }
    ]
)
print(f"\nAssistant: {response}")

print("\n📊 Observability: Inspecting the last search")
print("-" * 70)

if client.last_trace:
    print(f"\nSearch Trace:")
    for event in client.last_trace.events:
        print(f"  • {event.event_type}: {event.duration_ms:.1f}ms")
        if event.metadata:
            print(f"    Metadata: {event.metadata}")

    print(f"\nTotal search time: {client.last_trace.total_duration_ms:.1f}ms")

print("\n✅ Example complete!")
print("\n💡 Tips:")
print("  - Novita provides access to multiple open-source models through one API")
print("  - Use operation_mode='online' for faster startup in production")
print("  - Use operation_mode='lightweight' for prototyping")
print("  - Adjust salience_threshold to control memory filtering")
print("  - Check client.last_trace for performance insights")
print("\n💡 Novita Model Options:")
print("  - deepseek/deepseek-v3.2 - Best for coding and reasoning")
print("  - zai-org/glm-5 - Great for Chinese and English")
print("  - minimax/minimax-m2.5 - Strong multilingual support")

# Cleanup
client.close()
