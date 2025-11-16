"""
Claude (Anthropic) Provider Example

Demonstrates using MemLayer with Anthropic's Claude models.
Supports: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, etc.
"""

from memlayer.wrappers.claude import Claude
import os

# Check for API key
if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: Please set your ANTHROPIC_API_KEY environment variable.")
    print("export ANTHROPIC_API_KEY='sk-ant-...'")
    exit(1)

print("="*70)
print("MemLayer - CLAUDE EXAMPLE")
print("="*70)

# Initialize the memory-enhanced Claude client
client = Claude(
    # API key (optional - reads from ANTHROPIC_API_KEY env var by default)
    # api_key="sk-ant-...",
    
    # Model selection
    model="claude-3-5-sonnet-20241022",  # Options: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
    
    # Standard Claude parameters
    temperature=0.7,
    max_tokens=1024,
    
    # MemLayer settings
    storage_path="./claude_memories",
    user_id="demo_user",
    
    # Operation mode: "local" (default), "online", or "lightweight"
    # Note: "online" mode requires OPENAI_API_KEY for embeddings
    operation_mode="local",
)

print("\n📝 Conversation 1: Teaching Claude about yourself")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "Hello! I'm Marcus, a data scientist at DataFlow Inc."}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I'm currently analyzing customer churn data using machine learning."}
])
print(f"\nAssistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I prefer scikit-learn for classical ML and PyTorch for deep learning."}
])
print(f"\nAssistant: {response}")

# Wait for background consolidation
print("\n⏳ Waiting for memory consolidation...")
import time
time.sleep(3)

print("\n🔍 Conversation 2: Testing memory recall")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "What company do I work for and what's my role?"}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "What are my preferred ML frameworks?"}
])
print(f"\nAssistant: {response}")

print("\n📊 Observability: Inspecting the last search")
print("-" * 70)

if client.last_trace:
    print(f"\nSearch Trace:")
    for event in client.last_trace.events:
        print(f"  • {event.event_type}: {event.duration_ms:.1f}ms")
    
    print(f"\nTotal search time: {client.last_trace.total_duration_ms:.1f}ms")

print("\n✅ Example complete!")
print("\n💡 Tips:")
print("  - Claude excels at detailed, nuanced responses")
print("  - Use claude-3-5-haiku for faster responses")
print("  - Use claude-3-opus for most complex reasoning")
print("  - Set max_tokens to control response length")

# Cleanup
client.close()
