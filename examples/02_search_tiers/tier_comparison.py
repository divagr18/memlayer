"""
Visual Search Tier Comparison
==============================

This example creates a visual side-by-side comparison of all three search tiers
answering the same query with different levels of depth.
"""

import os
import time
from memlayer.wrappers.openai import OpenAI

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  Please set OPENAI_API_KEY environment variable")
    exit(1)

def print_box(title, content, width=70):
    """Print content in a nice box"""
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {title.center(width - 4)} ║")
    print("╠" + "═" * (width - 2) + "╣")
    for line in content.split('\n'):
        if line:
            print(f"║ {line.ljust(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")

print("\n" + "=" * 70)
print("VISUAL SEARCH TIER COMPARISON")
print("=" * 70)

# Initialize client
print("\n📦 Initializing client...")
client = OpenAI(
    model="gpt-4.1-mini",
    storage_path="./tier_comparison_memory",
    user_id="comparison_user"
)

# Seed with rich, interconnected data
print("\n📝 Seeding memory with interconnected information...")
data = [
    "I'm Dr. Sarah Chen, Chief Technology Officer at TechCorp.",
    "TechCorp is a SaaS company specializing in enterprise cloud solutions.",
    "I lead a team of 15 engineers across 3 offices: San Francisco, Austin, and Berlin.",
    "Our flagship product is DataFlow, a real-time data pipeline platform.",
    "DataFlow processes 10TB of data daily for Fortune 500 clients.",
    "I report directly to CEO Michael Rodriguez.",
    "Michael Rodriguez founded TechCorp in 2018 after selling his previous startup.",
    "My team includes senior engineers Lisa Park and Ahmed Hassan.",
    "Lisa Park specializes in distributed systems and has a PhD from Stanford.",
    "Ahmed Hassan is our security lead and worked at Google for 7 years.",
]

for item in data:
    client.chat([{"role": "user", "content": item}])
    print(f"   ✓ {item[:60]}...")

print("\n⏳ Waiting 4 seconds for consolidation...")
time.sleep(4)

# Define the test query
query = "Tell me about Dr. Sarah Chen and her work at TechCorp"

print("\n" + "=" * 70)
print(f"QUERY: {query}")
print("=" * 70)

# ============================================================================
# FAST TIER
# ============================================================================
print("\n⏱️  Running FAST tier search...")
start = time.time()
fast_response = client.chat([
    {"role": "user", "content": f"{query} (use fast tier)"}
])
fast_time = (time.time() - start) * 1000

fast_metadata = {}
if client.last_trace:
    for event in client.last_trace.events:
        if event.name == "vector_search":
            fast_metadata = event.metadata
            break

# ============================================================================
# BALANCED TIER
# ============================================================================
print("⏱️  Running BALANCED tier search...")
start = time.time()
balanced_response = client.chat([
    {"role": "user", "content": f"{query} (use balanced tier)"}
])
balanced_time = (time.time() - start) * 1000

balanced_metadata = {}
if client.last_trace:
    for event in client.last_trace.events:
        if event.name == "vector_search":
            balanced_metadata = event.metadata
            break

# ============================================================================
# DEEP TIER
# ============================================================================
print("⏱️  Running DEEP tier search...")
start = time.time()
deep_response = client.chat([
    {"role": "user", "content": f"{query} (use deep tier with full graph traversal)"}
])
deep_time = (time.time() - start) * 1000

deep_vector_metadata = {}
deep_graph_metadata = {}
if client.last_trace:
    for event in client.last_trace.events:
        if event.name == "vector_search":
            deep_vector_metadata = event.metadata
        elif event.name == "graph_search":
            deep_graph_metadata = event.metadata

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Fast tier results
print("\n🚀 FAST TIER")
print_box(
    "Configuration",
    f"Vector Results: {fast_metadata.get('results_found', 0)}\n"
    f"Graph Search: Disabled\n"
    f"Search Time: {fast_time:.2f}ms"
)
print_box(
    "Response",
    fast_response[:200] + "..." if len(fast_response) > 200 else fast_response
)

# Balanced tier results
print("\n⚖️  BALANCED TIER")
print_box(
    "Configuration",
    f"Vector Results: {balanced_metadata.get('results_found', 0)}\n"
    f"Graph Search: Disabled\n"
    f"Search Time: {balanced_time:.2f}ms"
)
print_box(
    "Response",
    balanced_response[:200] + "..." if len(balanced_response) > 200 else balanced_response
)

# Deep tier results
print("\n🔍 DEEP TIER")
graph_status = "Enabled" if deep_graph_metadata else "Enabled (no relationships found yet)"
print_box(
    "Configuration",
    f"Vector Results: {deep_vector_metadata.get('results_found', 0)}\n"
    f"Graph Search: {graph_status}\n"
    f"Entities Extracted: {deep_graph_metadata.get('extracted_entities', [])}\n"
    f"Relationships Found: {deep_graph_metadata.get('relationships_found', 0)}\n"
    f"Search Time: {deep_time:.2f}ms"
)
print_box(
    "Response",
    deep_response[:200] + "..." if len(deep_response) > 200 else deep_response
)

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

print(f"""
╔══════════════╦═══════════╦═══════════╦═══════════╗
║   Metric     ║   Fast    ║ Balanced  ║   Deep    ║
╠══════════════╬═══════════╬═══════════╬═══════════╣
║ Search Time  ║ {fast_time:>7.2f}ms ║ {balanced_time:>7.2f}ms ║ {deep_time:>7.2f}ms ║
║ Vector Hits  ║ {fast_metadata.get('results_found', 0):>9} ║ {balanced_metadata.get('results_found', 0):>9} ║ {deep_vector_metadata.get('results_found', 0):>9} ║
║ Graph Search ║     ✗     ║     ✗     ║     ✓     ║
║ Relationships║     -     ║     -     ║ {deep_graph_metadata.get('relationships_found', 0):>9} ║
╚══════════════╩═══════════╩═══════════╩═══════════╝
""")

# ============================================================================
# INSIGHTS
# ============================================================================
print("\n" + "=" * 70)
print("INSIGHTS")
print("=" * 70)

print(f"""
Speed Factor:
  • Fast is {fast_time/fast_time:.1f}x baseline
  • Balanced is {balanced_time/fast_time:.1f}x slower than Fast
  • Deep is {deep_time/fast_time:.1f}x slower than Fast

Information Depth:
  • Fast retrieved {fast_metadata.get('results_found', 0)} memories
  • Balanced retrieved {balanced_metadata.get('results_found', 0)} memories (+{balanced_metadata.get('results_found', 0) - fast_metadata.get('results_found', 0)})
  • Deep retrieved {deep_vector_metadata.get('results_found', 0)} memories + {deep_graph_metadata.get('relationships_found', 0)} relationships

When to Use Each:
  🚀 FAST: Real-time chat, quick lookups, latency-critical apps
  ⚖️  BALANCED: General queries, everyday conversation (default)
  🔍 DEEP: Research, complex questions, relationship discovery

Quality vs Speed Trade-off:
  • Fast sacrifices completeness for speed
  • Balanced provides good middle ground
  • Deep prioritizes comprehensive answers over speed
""")

# ============================================================================
# GRAPH SEARCH DETAILS
# ============================================================================
if deep_graph_metadata and deep_graph_metadata.get('relationships_found', 0) > 0:
    print("\n" + "=" * 70)
    print("GRAPH SEARCH DETAILS (Deep Tier Only)")
    print("=" * 70)
    
    print(f"""
Entity Extraction:
  The LLM identified these key entities in the query:
  {deep_graph_metadata.get('extracted_entities', [])}

Knowledge Graph Traversal:
  Found {deep_graph_metadata.get('relationships_found', 0)} relationships by traversing
  the knowledge graph, revealing connections between entities.

Example relationships might include:
  • (Person) Dr. Sarah Chen --[works at]--> (Company) TechCorp
  • (Person) Dr. Sarah Chen --[leads]--> (Team) Engineering Team
  • (Product) DataFlow --[belongs to]--> (Company) TechCorp
  • (Person) Lisa Park --[reports to]--> (Person) Dr. Sarah Chen
    """)
else:
    print("\n" + "=" * 70)
    print("NOTE: Graph Relationships")
    print("=" * 70)
    print("""
⚠️  The graph search is enabled but no relationships were found yet.

This is normal on the first run because:
  1. Knowledge graph building happens in a background thread
  2. Entity/relationship extraction takes a few seconds
  3. The graph may still be consolidating

💡 To see graph traversal in action:
  1. Wait 5-10 seconds after the first run
  2. Run this script again
  3. The deep tier will now show relationship traversal
    """)

print("\n" + "=" * 70)
print("✅ Comparison Complete!")
print("=" * 70)
print("\n💡 Try running this script multiple times to see how the graph")
print("   becomes richer with relationships over time!")
