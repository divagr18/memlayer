"""
Google Gemini Provider Example

Demonstrates using MemLayer with Google's Gemini models.
Supports: gemini-2.5-flash-lite, gemini-2.5-pro, gemini-2.5-flash, etc.
"""

from memlayer.wrappers.gemini import Gemini
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: Please set your GOOGLE_API_KEY environment variable.")
    print("export GOOGLE_API_KEY='...'")
    exit(1)

print("="*70)
print("MemLayer - GOOGLE GEMINI EXAMPLE")
print("="*70)

# Initialize the memory-enhanced Gemini client
client = Gemini(
    # API key (optional - reads from GOOGLE_API_KEY env var by default)
    # api_key="...",
    
    # Model selection
    model="gemini-2.5-flash",  # Options: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash
    
    # Standard Gemini parameters
    temperature=0.7,
    
    # MemLayer settings
    storage_path="./gemini_memories",
    user_id="demo_user",
    
    # Operation mode: "local" (default), "online", or "lightweight"
    # Note: "online" mode requires OPENAI_API_KEY for embeddings
    operation_mode="local",

)

print("\n📝 Conversation 1: Teaching Gemini about yourself")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "Hi! I'm Alex, a product manager at InnovateLabs."}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I'm leading the development of a mobile health app called FitTrack."}
])
print(f"\nAssistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "Our tech stack includes Flutter for mobile and Firebase for backend."}
])
print(f"\nAssistant: {response}")

# Wait for background consolidation
print("\n⏳ Waiting for memory consolidation...")
import time
time.sleep(3)

print("\n🔍 Conversation 2: Testing memory recall")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "What product am I working on?"}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "What technologies does FitTrack use?"}
])
print(f"\nAssistant: {response}")

print("\n📊 Observability: Inspecting the last search")
print("-" * 70)

if client.last_trace:
    print(f"\nSearch Trace:")
    for event in client.last_trace.events:
        print(f"  • {event.name}: {event.duration_ms:.1f}ms")
    
    print(f"\nTotal search time: {client.last_trace.total_duration_ms:.1f}ms")

print("\n✅ Example complete!")
print("\n💡 Tips:")
print("  - Gemini 2.5 Flash is extremely fast for most tasks")
print("  - Gemini 2.5 Pro offers best reasoning capabilities")
print("  - Free tier available for development/testing")
print("  - Supports multimodal inputs (text, images, audio)")

# Cleanup
client.close()
