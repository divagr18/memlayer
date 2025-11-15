"""
Balanced Tier Search Example
=============================

The BALANCED tier provides a good mix of accuracy and performance (<500ms).
This is the default tier for most queries and handles general conversation well.

Characteristics:
- Vector search only (no graph traversal)
- Returns top 5 most relevant memories
- Target latency: <500ms
- Use case: General questions, standard conversation, most queries
"""

import os
import time
from memory_bank.wrappers.openai import OpenAI

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  Please set OPENAI_API_KEY environment variable")
    exit(1)

print("⚖️  Balanced Tier Search Example")
print("=" * 60)

# Initialize client
client = OpenAI(
    model="gpt-4o-mini",
    storage_path="./balanced_tier_memory",
    user_id="balanced_demo_user"
)

# Store multiple related facts
print("\n📝 Storing information about a project...")
facts = [
    "I'm working on the CloudSync project, a file synchronization service.",
    "CloudSync uses AWS S3 for storage and supports real-time collaboration.",
    "The project started in March 2024 and we have 100,000 active users.",
    "Our main competitors are Dropbox and Google Drive.",
    "We just launched a mobile app for iOS and Android last month.",
    "The team consists of 8 engineers and 2 designers.",
]

for fact in facts:
    print(f"   • {fact[:50]}...")
    client.chat([{"role": "user", "content": fact}])

print("\n⏳ Waiting for consolidation...")
time.sleep(3)

# Query using balanced tier (default)
print("\n🔍 Balanced search query: 'Tell me about the CloudSync project'")

response = client.chat([
    {"role": "user", "content": "Tell me about the CloudSync project"}
])

print(f"\n✅ Response: {response}")

if client.last_trace:
    print("\n📊 Performance metrics:")
    total_duration = 0
    for event in client.last_trace.events:
        total_duration += event.duration_ms
        if event.name == "vector_search":
            print(f"   - Search tier: {event.metadata.get('tier', 'unknown')}")
            print(f"   - Results retrieved: {event.metadata.get('results_found', 0)}")
            print(f"   - Vector search duration: {event.duration_ms:.2f}ms")
    print(f"   - Total search time: {total_duration:.2f}ms")

print("\n💡 Balanced tier is perfect for:")
print("   • General conversational queries")
print("   • Multi-fact recall")
print("   • Standard Q&A applications")
print("   • Most production use cases (default choice)")
print("\n📌 Note: This is the default tier when you don't specify one.")
