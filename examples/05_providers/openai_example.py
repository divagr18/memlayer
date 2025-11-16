"""
OpenAI Provider Example

Demonstrates using MemLayer with OpenAI's GPT models.
Supports: gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo, etc.
"""

from memlayer.wrappers.openai import OpenAI
import os

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable.")
    print("export OPENAI_API_KEY='sk-...'")
    exit(1)

print("="*70)
print("MemLayer - OPENAI EXAMPLE")
print("="*70)

# Initialize the memory-enhanced OpenAI client
client = OpenAI(
    # API key (optional - reads from OPENAI_API_KEY env var by default)
    # api_key="sk-...",
    
    # Model selection
    model="gpt-4o-mini",  # Options: gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo
    
    # Standard OpenAI parameters
    temperature=0.7,
    
    # MemLayer settings
    storage_path="./openai_memories",
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

response = client.chat(messages=[
    {"role": "user", "content": "Hi! My name is Sarah and I'm a software engineer at TechCorp."}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I'm working on Project Phoenix, which uses Python and React."}
])
print(f"\nAssistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "My favorite programming language is Python and I prefer tabs over spaces."}
])
print(f"\nAssistant: {response}")

# Wait for background consolidation
print("\n⏳ Waiting for memory consolidation...")
import time
time.sleep(3)

print("\n🔍 Conversation 2: Testing memory recall")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "What do you know about me?"}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "What project am I working on and what technologies does it use?"}
])
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
print("  - Use salience_mode='online' for faster startup in production")
print("  - Use salience_mode='lightweight' for prototyping")
print("  - Adjust salience_threshold to control memory filtering")
print("  - Check client.last_trace for performance insights")

# Cleanup
client.close()
