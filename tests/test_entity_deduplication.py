"""Test entity deduplication in the knowledge graph."""

from memlayer.storage.networkx import NetworkXStorage
import shutil
from pathlib import Path

# Clean up any existing test graph
test_path = Path("test_dedup_memory")
if test_path.exists():
    shutil.rmtree(test_path)
test_path.mkdir()

print("=" * 70)
print("ENTITY DEDUPLICATION TEST")
print("=" * 70)

# Create storage
graph_storage = NetworkXStorage(str(test_path))

print("\n1. Adding 'Dr. Watson' (Person)")
graph_storage.add_entity("Dr. Watson", "Person")
graph_storage.add_relationship("Dr. Watson", "works at", "MIT")

print("\n2. Adding 'Dr. Emma Watson' (Person) - should merge with Dr. Watson")
graph_storage.add_entity("Dr. Emma Watson", "Person")
graph_storage.add_relationship("Dr. Emma Watson", "researches", "machine learning")

print("\n3. Adding 'Emma Watson' (Person) - should also merge")
graph_storage.add_relationship("Emma Watson", "collaborates with", "Professor Chen")

print("\n4. Adding 'Watson' (Person) - should also merge")
graph_storage.add_relationship("Watson", "leads", "Neural Architecture Search")

print("\n" + "=" * 70)
print("FINAL GRAPH STATE:")
print("=" * 70)

print(f"\nTotal nodes: {graph_storage.graph.number_of_nodes()}")
print(f"Total edges: {graph_storage.graph.number_of_edges()}")

print("\n" + "=" * 70)
print("ALL PERSON NODES:")
print("=" * 70)
for node, data in graph_storage.graph.nodes(data=True):
    if data.get('type') == 'Person':
        print(f"  - {node}")

print("\n" + "=" * 70)
print("ALL RELATIONSHIPS:")
print("=" * 70)
for u, v, data in graph_storage.graph.edges(data=True):
    rel_type = data.get('type', 'related to')
    print(f"  {u} --[{rel_type}]--> {v}")

print("\n" + "=" * 70)
print("TEST RESULTS:")
print("=" * 70)

# Count how many Watson-related nodes exist
watson_nodes = [n for n in graph_storage.graph.nodes() 
                if 'watson' in n.lower() and 
                graph_storage.graph.nodes[n].get('type') == 'Person']

print(f"\nNumber of 'Watson' person nodes: {len(watson_nodes)}")
if len(watson_nodes) == 1:
    print(f"✅ SUCCESS: All Watson entities merged into '{watson_nodes[0]}'")
    print(f"   This node has {graph_storage.graph.degree(watson_nodes[0])} relationships")
else:
    print(f"❌ FAIL: Expected 1 node, got {len(watson_nodes)}: {watson_nodes}")

# Cleanup
shutil.rmtree(test_path)
