"""Test 2-hop traversal to see what we get."""

from memory_bank.storage.networkx import NetworkXStorage

graph_storage = NetworkXStorage("deep_tier_memory")

print("=" * 70)
print("TESTING 2-HOP TRAVERSAL FROM DR. WATSON")
print("=" * 70)

# Try different depths
for depth in [1, 2, 3]:
    rels = graph_storage.get_subgraph_context("Dr. Watson", depth=depth)
    print(f"\nDepth {depth}: {len(rels)} relationships")
    for rel in rels[:10]:  # Show first 10
        print(f"  {rel}")
    if len(rels) > 10:
        print(f"  ... and {len(rels) - 10} more")

# Also check Dr. Emma Watson
print("\n" + "=" * 70)
print("TESTING FROM 'DR. EMMA WATSON'")
print("=" * 70)

if graph_storage.graph.has_node("Dr. Emma Watson"):
    for depth in [1, 2]:
        rels = graph_storage.get_subgraph_context("Dr. Emma Watson", depth=depth)
        print(f"\nDepth {depth}: {len(rels)} relationships")
        for rel in rels[:10]:
            print(f"  {rel}")
else:
    print("Node 'Dr. Emma Watson' not found in graph")
    
# Check what nodes have "Watson" in them
print("\n" + "=" * 70)
print("ALL NODES CONTAINING 'WATSON':")
print("=" * 70)
for node in graph_storage.graph.nodes():
    if 'watson' in node.lower():
        print(f"  - {node}")
