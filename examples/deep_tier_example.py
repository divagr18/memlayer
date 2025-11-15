"""
Deep Tier Search Example
=========================

The DEEP tier provides comprehensive search with knowledge graph reasoning (<2s).
It combines vector search with entity extraction and graph traversal for complex queries.

Characteristics:
- Vector search: Returns top 10 most relevant memories
- Graph search: Extracts entities and traverses relationships (1-hop)
- Target latency: <2s
- Use case: Complex questions, relationship discovery, multi-hop reasoning

How Deep Search Works:
1. Performs semantic vector search
2. Extracts key entities from the query using LLM
3. Traverses the knowledge graph for each entity
4. Combines vector results with graph relationships
5. LLM synthesizes comprehensive answer
"""

import os
import time
from memory_bank.wrappers.openai import OpenAI

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  Please set OPENAI_API_KEY environment variable")
    exit(1)

print("🔍 Deep Tier Search Example")
print("=" * 60)

# Initialize client
client = OpenAI(
    model="gpt-4o-mini",
    storage_path="./deep_tier_memory",
    user_id="deep_demo_user"
)

# Store interconnected information
print("\n📝 Storing interconnected information...")
conversations = [
    "I'm Dr. Emma Watson, a machine learning researcher at MIT.",
    "I'm leading the Neural Architecture Search project with Professor James Chen.",
    "Professor James Chen is the head of the AI Lab and has published over 100 papers.",
    "The Neural Architecture Search project aims to automate neural network design.",
    "We're collaborating with Google Brain and have received a $2M grant from NSF.",
    "My PhD student Alex Kim is working on the automated hyperparameter tuning component.",
    "Alex Kim previously worked at DeepMind for 3 years before joining MIT.",
    "The project started in September 2023 and we've already published 5 papers.",
]

for conv in conversations:
    print(f"   • {conv[:70]}...")
    client.chat([{"role": "user", "content": conv}])

print("\n⏳ Waiting 5 seconds for consolidation and graph building...")
time.sleep(5)

# Complex query using deep tier
print("\n🔍 Deep search query: 'Tell me about Dr. Emma Watson and all her connections'")
print("   (Explicitly requesting deep tier for comprehensive results)")

response = client.chat([
    {"role": "user", "content": "Tell me everything about Dr. Emma Watson and all her connections. Use deep search to find all relationships."}
])

print(f"\n✅ Response: {response}")

if client.last_trace:
    print("\n📊 Deep Search Analytics:")
    print("\n   Vector Search:")
    for event in client.last_trace.events:
        if event.name == "vector_search":
            print(f"      - Tier: {event.metadata.get('tier', 'unknown')}")
            print(f"      - Results retrieved: {event.metadata.get('results_found', 0)}")
            print(f"      - Duration: {event.duration_ms:.2f}ms")
    
    print("\n   Graph Search:")
    graph_event = next((e for e in client.last_trace.events if e.name == 'graph_search'), None)
    if graph_event:
        entities = graph_event.metadata.get('extracted_entities', [])
        matched = graph_event.metadata.get('matched_entities', [])
        relationships = graph_event.metadata.get('relationships_found', 0)
        nodes_traversed = graph_event.metadata.get('nodes_traversed', 0)
        print(f"      - Entities extracted: {entities}")
        if matched:
            print(f"      - Matched in graph:")
            for m in matched:
                print(f"         • {m}")
        print(f"      - Nodes traversed: {nodes_traversed}")
        print(f"      - Relationships found: {relationships}")
        print(f"      - Duration: {graph_event.duration_ms:.2f}ms")
    else:
        print("      - No graph search performed (graph may be empty)")
    
    total = sum(e.duration_ms for e in client.last_trace.events)
    print(f"\n   Total search time: {total:.2f}ms")

print("\n💡 Deep tier is perfect for:")
print("   • Complex relationship queries")
print("   • Multi-hop reasoning ('Who knows X who worked with Y?')")
print("   • Research and analysis tasks")
print("   • Finding hidden connections")
print("   • Comprehensive information gathering")

print("\n🔄 Try running this example multiple times:")
print("   • First run: May not find graph relationships (graph building in background)")
print("   • Second run: Should show rich graph traversal results")

print("\n📊 Deep vs Balanced comparison:")
print("   • Deep retrieves 2x more vector results (10 vs 5)")
print("   • Deep adds graph-based relationship discovery")
print("   • Deep takes ~4x longer but provides comprehensive answers")
