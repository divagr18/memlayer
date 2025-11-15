"""
Test hybrid search with graph traversal for deep tier
"""
import os
from memory_bank.wrappers.openai import OpenAI

# Create client (requires OpenAI API key)
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  OPENAI_API_KEY not set. Skipping test.")
    exit(0)

client = OpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    storage_path="./test_hybrid_memory",
    user_id="test_user"
)

print("✅ OpenAI client created with lazy loading")
print(f"   Model: {client.model}")
print(f"   Storage path: {client.storage_path}")

# Test 1: Simple conversation (won't trigger memory search)
print("\n📝 Test 1: Simple conversation")
response = client.chat([
    {"role": "user", "content": "Hello! My name is John and I work on Project Phoenix."}
])
print(f"Response: {response}")

# Test 2: Memory search with fast tier (vector only)
print("\n🔍 Test 2: Fast tier search (vector only)")
response = client.chat([
    {"role": "user", "content": "What project do I work on?"}
])
print(f"Response: {response}")

# Test 3: Deep tier search (should trigger entity extraction + graph traversal)
print("\n🔍 Test 3: Deep tier search (vector + graph)")
response = client.chat([
    {"role": "user", "content": "Tell me everything you know about Project Phoenix using deep search."}
])
print(f"Response: {response}")

# Check if trace shows graph search was performed
if client.last_trace:
    print("\n📊 Trace information:")
    for event in client.last_trace.events:
        print(f"   - {event.event_type}: {event.duration_ms:.2f}ms")
        if event.metadata:
            print(f"     Metadata: {event.metadata}")

print("\n✅ Hybrid search test complete!")
print("\nNote: The graph will be populated during consolidation (background thread).")
print("Run this test again after a few seconds to see graph traversal in action.")
