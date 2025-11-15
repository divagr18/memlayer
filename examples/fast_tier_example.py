"""
Fast Tier Search Example
=========================

The FAST tier is optimized for quick lookups with minimal latency (<100ms).
Perfect for real-time applications, chatbots, and simple recall queries.

Characteristics:
- Vector search only (no graph traversal)
- Returns top 2 most relevant memories
- Target latency: <100ms
- Use case: "What's my name?", "What's the status?", simple factual recalls
"""

import os
from memory_bank.wrappers.openai import OpenAI

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  Please set OPENAI_API_KEY environment variable")
    exit(1)

print("🚀 Fast Tier Search Example")
print("=" * 60)

# Initialize client
client = OpenAI(
    model="gpt-4o-mini",
    storage_path="./fast_tier_memory",
    user_id="fast_demo_user"
)

# Store some quick facts
print("\n📝 Storing information...")
client.chat([
    {"role": "user", "content": "My name is Sarah and I work in the Marketing department."}
])
client.chat([
    {"role": "user", "content": "I prefer morning meetings and coffee without sugar."}
])

import time
print("⏳ Waiting for consolidation...")
time.sleep(2)

# Quick lookup using fast tier
print("\n🔍 Fast search query: 'What's my name?'")
print("   (The LLM will automatically use fast tier for simple queries)")

response = client.chat([
    {"role": "user", "content": "What's my name?"}
])

print(f"\n✅ Response: {response}")

if client.last_trace:
    print("\n📊 Performance metrics:")
    for event in client.last_trace.events:
        if event.name == "vector_search":
            print(f"   - Search tier: {event.metadata.get('tier', 'unknown')}")
            print(f"   - Results retrieved: {event.metadata.get('results_found', 0)}")
            print(f"   - Duration: {event.duration_ms:.2f}ms")

print("\n💡 Fast tier is perfect for:")
print("   • Real-time chat applications")
print("   • Quick factual lookups")
print("   • Latency-sensitive scenarios")
print("   • Simple yes/no questions")
