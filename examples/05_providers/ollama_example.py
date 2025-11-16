"""
Ollama (Local) Provider Example

Demonstrates using MemLayer with Ollama for local LLM inference.
Supports: llama3.2, qwen3, mistral, phi3, and any other Ollama models.

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull a model: ollama pull qwen3:1.7b
3. Start Ollama server (usually runs automatically)
"""

from memlayer.wrappers.ollama import Ollama

print("="*70)
print("MemLayer - OLLAMA (LOCAL) EXAMPLE")
print("="*70)

# Initialize the memory-enhanced Ollama client
client = Ollama(
    # Ollama server URL (default: http://localhost:11434)
    host="http://localhost:11434",
    
    # Model selection (must be already pulled via 'ollama pull')
    model="qwen3:1.7b",  # Options: llama3.2, qwen3, mistral, phi3, etc.
    
    # Standard Ollama parameters
    temperature=0.7,
    
    # MemLayer settings
    storage_path="./ollama_memories",
    user_id="demo_user",
    
    # Operation mode: "local" (default), "online", or "lightweight"
    # For offline/local use, use "local" or "lightweight"
    operation_mode="local",
)

print("\n📝 Conversation 1: Teaching the local LLM about yourself")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "Hello! My name is Jordan and I'm a cybersecurity analyst."}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I work on threat detection systems using Python and network analysis."}
])
print(f"\nAssistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "My favorite tools are Wireshark, Metasploit, and Burp Suite."}
])
print(f"\nAssistant: {response}")

# Wait for background consolidation
print("\n⏳ Waiting for memory consolidation...")
import time
time.sleep(3)

print("\n🔍 Conversation 2: Testing memory recall")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "What's my profession?"}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "What tools do I use in my work?"}
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
print("\n💡 Tips for Ollama:")
print("  - Run entirely offline - no API costs!")
print("  - Use lightweight models like qwen3:1.7b for fast responses")
print("  - Use larger models like llama3.2:8b for better quality")
print("  - Perfect for privacy-sensitive applications")
print("  - Check ollama.ai/library for available models")
print("\n📦 Quick Ollama Setup:")
print("  1. Install: curl https://ollama.ai/install.sh | sh")
print("  2. Pull model: ollama pull qwen3:1.7b")
print("  3. Run this script!")

# Cleanup
client.close()
