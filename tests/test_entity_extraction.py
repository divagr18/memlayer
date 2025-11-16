"""Test what entities are being extracted from queries."""

from memlayer.wrappers.openai import OpenAI
import os

# Initialize client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4.1-mini",
    storage_path="test_extraction"
)

test_queries = [
    "Tell me about Dr. Emma Watson and all her connections",
    "What does Dr. Watson work on?",
    "Tell me about the Neural Architecture Search project",
    "Who is working with Professor James Chen?",
    "What is Alex Kim doing?"
]

print("=" * 70)
print("ENTITY EXTRACTION TEST")
print("=" * 70)

for query in test_queries:
    print(f"\n📝 Query: '{query}'")
    entities = client.extract_query_entities(query)
    print(f"   Extracted: {entities}")
