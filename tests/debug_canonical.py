"""Debug the canonical entity finder."""

from memlayer.storage.networkx import NetworkXStorage
import shutil
from pathlib import Path

# Clean up
test_path = Path("test_dedup_debug")
if test_path.exists():
    shutil.rmtree(test_path)
test_path.mkdir()

graph_storage = NetworkXStorage(str(test_path))

# Add first entity
print("Adding 'Dr. Watson'...")
canonical1 = graph_storage.add_entity("Dr. Watson", "Person")
print(f"  Canonical name: '{canonical1}'")
print(f"  Nodes in graph: {list(graph_storage.graph.nodes())}")

# Now add a longer version
print("\nAdding 'Dr. Emma Watson'...")
canonical2 = graph_storage.add_entity("Dr. Emma Watson", "Person")
print(f"  Canonical name: '{canonical2}'")
print(f"  Nodes in graph: {list(graph_storage.graph.nodes())}")

# Test the finder directly
print("\n" + "=" * 70)
print("DIRECT TEST OF _find_canonical_entity:")
print("=" * 70)

test_names = ["Dr. Watson", "Dr. Emma Watson", "Emma Watson", "Watson", "Dr Emma Watson"]
for test_name in test_names:
    canonical = graph_storage._find_canonical_entity(test_name, "Person", 0.85)
    print(f"  '{test_name}' -> '{canonical}'")

# Cleanup
shutil.rmtree(test_path)
