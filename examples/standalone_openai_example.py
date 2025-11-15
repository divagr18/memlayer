"""
Example: Using the standalone memory-enhanced OpenAI client

This demonstrates the new, simpler way to use Memory Bank with OpenAI.
No need to create a Memory object first - just import and use!
"""

from memory_bank import OpenAI

# Initialize the memory-enhanced OpenAI client
# It handles everything: storage, embeddings, knowledge graph, etc.
client = OpenAI(
    api_key="your-openai-api-key-here",  # or set OPENAI_API_KEY env var
    model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo", etc.
    temperature=0.7,
    storage_path="./my_chat_memories",
    user_id="user_123"
)

# Example conversation 1: Teaching the AI about preferences
print("=" * 60)
print("Teaching the AI about your preferences...")
print("=" * 60)

response1 = client.chat(messages=[
    {"role": "user", "content": "Hi! I'm working on a project called 'Phoenix' and I prefer Python for backend development."}
])
print(f"Assistant: {response1}\n")

# Example conversation 2: The AI doesn't need memory yet (fast path)
print("=" * 60)
print("Simple question (no memory needed)...")
print("=" * 60)

response2 = client.chat(messages=[
    {"role": "user", "content": "What's 25 * 4?"}
])
print(f"Assistant: {response2}\n")

# Example conversation 3: AI will automatically search memory
print("=" * 60)
print("Asking about something from earlier (AI will search memory)...")
print("=" * 60)

response3 = client.chat(messages=[
    {"role": "user", "content": "What programming language do I prefer for backend?"}
])
print(f"Assistant: {response3}\n")

# You can also customize per-request
print("=" * 60)
print("Using custom settings for one request...")
print("=" * 60)

response4 = client.chat(
    messages=[
        {"role": "user", "content": "Write me a haiku about coding"}
    ],
    temperature=1.2,  # More creative for this one request
    max_tokens=100
)
print(f"Assistant: {response4}\n")

print("=" * 60)
print("All conversations have been automatically saved to memory!")
print(f"Check {client.vector_storage.db_path} for stored memories")
print(f"Check {client.graph_storage.graph_path} for the knowledge graph")
print("=" * 60)
