"""Debug script to inspect what's actually in the knowledge graph."""

from memory_bank.storage.networkx import NetworkXStorage

# Load the graph from the deep tier example
graph_storage = NetworkXStorage("deep_tier_memory")

print("=" * 70)
print("KNOWLEDGE GRAPH DEBUG")
print("=" * 70)

print(f"\nTotal nodes: {graph_storage.graph.number_of_nodes()}")
print(f"Total edges: {graph_storage.graph.number_of_edges()}")

print("\n" + "=" * 70)
print("ALL NODES IN GRAPH:")
print("=" * 70)
for node in sorted(graph_storage.graph.nodes()):
    node_type = graph_storage.graph.nodes[node].get('type', 'Unknown')
    print(f"  - {node} (type: {node_type})")

print("\n" + "=" * 70)
print("ALL RELATIONSHIPS:")
print("=" * 70)
for u, v, data in graph_storage.graph.edges(data=True):
    rel_type = data.get('type', 'related to')
    print(f"  {u} --[{rel_type}]--> {v}")

print("\n" + "=" * 70)
print("TEST FUZZY MATCHING:")
print("=" * 70)

test_queries = [
    "Dr. Emma Watson",
    "Emma Watson", 
    "Professor James Chen",
    "James Chen",
    "Alex Kim",
    "Google Brain",
    "MIT"
]

for query in test_queries:
    matches = graph_storage.find_matching_nodes(query, threshold=0.6)
    print(f"\nQuery: '{query}'")
    print(f"  Matches: {matches}")
    
    if matches:
        # Show what relationships we'd get
        rels = graph_storage.get_subgraph_context(matches[0], depth=1)
        print(f"  Relationships ({len(rels)}):")
        for rel in rels[:3]:  # Show first 3
            print(f"    - {rel}")
