"""
MemLayer Search Tiers Demo
==============================

This example demonstrates the three search tiers available in MemLayer:
1. FAST - Quick lookups with minimal latency (<100ms)
2. BALANCED - Standard search with good performance (<500ms)
3. DEEP - Comprehensive search with knowledge graph reasoning (<2s)

Each tier has different characteristics and use cases.
"""

import os
import time
from memlayer.wrappers.openai import OpenAI

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  Please set OPENAI_API_KEY environment variable")
    print("   Example: export OPENAI_API_KEY='your-key-here'")
    exit(1)

print("=" * 70)
print("MemLayer Search Tiers Demo")
print("=" * 70)

# Initialize the client
print("\n📦 Initializing OpenAI client with memory...")
client = OpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    storage_path="./search_tiers_demo_memory",
    user_id="demo_user"
)
print(f"✅ Client ready (Model: {client.model})")

# ============================================================================
# PHASE 1: Seed the memory with some information
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 1: Seeding Memory with Information")
print("=" * 70)

conversations = [
    "My name is Alice and I'm a software engineer working on Project Phoenix in the London office.",
    "Project Phoenix is a cloud migration initiative that started in January 2024.",
    "The London office has 50 employees and is located at 123 Tech Street.",
    "I'm leading a team of 5 developers: Bob, Charlie, Diana, Eve, and Frank.",
    "We recently completed the database migration to PostgreSQL with zero downtime.",
    "Our next milestone is to migrate the API gateway, scheduled for December 2025.",
]

for i, info in enumerate(conversations, 1):
    print(f"\n💬 Conversation {i}: {info[:60]}...")
    response = client.chat([
        {"role": "user", "content": info}
    ])
    print(f"   Response: {response[:100]}...")

print("\n⏳ Waiting 3 seconds for background consolidation to complete...")
time.sleep(3)

# ============================================================================
# PHASE 2: Demonstrate FAST tier
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2: FAST Tier Search")
print("=" * 70)
print("""
🚀 FAST TIER
- Purpose: Quick lookups for simple queries
- Vector search: top_k=2 (returns 2 most relevant memories)
- Graph search: Disabled
- Target latency: <100ms
- Use case: Chatbots, real-time applications, simple recall
""")

print("\n📝 Query: 'What's my name?'")
start_time = time.time()
response = client.chat([
    {"role": "user", "content": "What's my name? Use fast search tier."}
])
elapsed = (time.time() - start_time) * 1000
print(f"\n✅ Response: {response}")
print(f"⏱️  Total time: {elapsed:.2f}ms")

if client.last_trace:
    print("\n📊 Trace Details:")
    for event in client.last_trace.events:
        print(f"   - {event.name}: {event.duration_ms:.2f}ms")
        if event.metadata:
            print(f"     {event.metadata}")

# ============================================================================
# PHASE 3: Demonstrate BALANCED tier
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3: BALANCED Tier Search")
print("=" * 70)
print("""
⚖️ BALANCED TIER
- Purpose: Standard search with good accuracy/performance balance
- Vector search: top_k=5 (returns 5 most relevant memories)
- Graph search: Disabled
- Target latency: <500ms
- Use case: Most conversational queries, general question-answering
""")

print("\n📝 Query: 'Tell me about Project Phoenix'")
start_time = time.time()
response = client.chat([
    {"role": "user", "content": "Tell me about Project Phoenix. Use balanced search tier."}
])
elapsed = (time.time() - start_time) * 1000
print(f"\n✅ Response: {response}")
print(f"⏱️  Total time: {elapsed:.2f}ms")

if client.last_trace:
    print("\n📊 Trace Details:")
    for event in client.last_trace.events:
        print(f"   - {event.name}: {event.duration_ms:.2f}ms")
        if event.metadata:
            print(f"     {event.metadata}")

# ============================================================================
# PHASE 4: Demonstrate DEEP tier
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 4: DEEP Tier Search")
print("=" * 70)
print("""
🔍 DEEP TIER
- Purpose: Comprehensive search with knowledge graph reasoning
- Vector search: top_k=10 (returns 10 most relevant memories)
- Graph search: Enabled (extracts entities and traverses relationships)
- Target latency: <2s
- Use case: Complex queries, multi-hop reasoning, relationship discovery

How it works:
1. Performs vector search (retrieves semantic matches)
2. Extracts key entities from the query using LLM
3. Traverses the knowledge graph for each entity (1-hop radius)
4. Combines vector results with graph relationships
5. LLM synthesizes a comprehensive answer
""")

print("\n📝 Query: 'Tell me everything about Alice and her work'")
start_time = time.time()
response = client.chat([
    {"role": "user", "content": "Tell me everything about Alice and her work. Use deep search tier to find all connections."}
])
elapsed = (time.time() - start_time) * 1000
print(f"\n✅ Response: {response}")
print(f"⏱️  Total time: {elapsed:.2f}ms")

if client.last_trace:
    print("\n📊 Trace Details:")
    total_duration = 0
    for event in client.last_trace.events:
        print(f"   - {event.name}: {event.duration_ms:.2f}ms")
        total_duration += event.duration_ms
        if event.metadata:
            print(f"     {event.metadata}")
    print(f"\n   Total traced duration: {total_duration:.2f}ms")

# ============================================================================
# PHASE 5: Side-by-side comparison
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 5: Side-by-Side Comparison")
print("=" * 70)

query = "What team members work with Alice?"

print(f"\n📝 Same query across all tiers: '{query}'")
print("\n" + "-" * 70)

# Fast tier
print("\n🚀 FAST TIER:")
start = time.time()
fast_response = client.chat([
    {"role": "user", "content": f"{query} (fast tier)"}
])
fast_time = (time.time() - start) * 1000
print(f"Response: {fast_response}")
print(f"Time: {fast_time:.2f}ms")
if client.last_trace:
    fast_vector_results = next((e.metadata.get('results_found', 0) for e in client.last_trace.events if e.name == 'vector_search'), 0)
    print(f"Vector results: {fast_vector_results}")

# Balanced tier
print("\n⚖️ BALANCED TIER:")
start = time.time()
balanced_response = client.chat([
    {"role": "user", "content": f"{query} (balanced tier)"}
])
balanced_time = (time.time() - start) * 1000
print(f"Response: {balanced_response}")
print(f"Time: {balanced_time:.2f}ms")
if client.last_trace:
    balanced_vector_results = next((e.metadata.get('results_found', 0) for e in client.last_trace.events if e.name == 'vector_search'), 0)
    print(f"Vector results: {balanced_vector_results}")

# Deep tier
print("\n🔍 DEEP TIER:")
start = time.time()
deep_response = client.chat([
    {"role": "user", "content": f"{query} (deep tier with graph traversal)"}
])
deep_time = (time.time() - start) * 1000
print(f"Response: {deep_response}")
print(f"Time: {deep_time:.2f}ms")
if client.last_trace:
    deep_vector_results = next((e.metadata.get('results_found', 0) for e in client.last_trace.events if e.name == 'vector_search'), 0)
    print(f"Vector results: {deep_vector_results}")
    graph_event = next((e for e in client.last_trace.events if e.name == 'graph_search'), None)
    if graph_event:
        print(f"Entities extracted: {graph_event.metadata.get('extracted_entities', [])}")
        print(f"Graph relationships found: {graph_event.metadata.get('relationships_found', 0)}")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"""
Performance Comparison:
  Fast:     {fast_time:>8.2f}ms (lightest, 2 results)
  Balanced: {balanced_time:>8.2f}ms (standard, 5 results)
  Deep:     {deep_time:>8.2f}ms (comprehensive, 10 results + graph)

Choosing the Right Tier:
  - Use FAST for: Real-time chat, simple lookups, latency-sensitive apps
  - Use BALANCED for: General conversation, most queries (default)
  - Use DEEP for: Complex questions, relationship discovery, research queries

The deep tier provides the most comprehensive results by combining:
  ✅ Semantic vector search (similarity matching)
  ✅ Knowledge graph traversal (relationship reasoning)
  ✅ Entity extraction (contextual understanding)
""")

print("\n🎉 Demo complete! Check the './search_tiers_demo_memory' folder for stored data.")
print("\n💡 Tip: The LLM automatically chooses the search tier based on query complexity,")
print("   but you can explicitly request a tier by mentioning it in your query.")
