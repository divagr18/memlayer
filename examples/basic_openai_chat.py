import openai
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import memory_bank
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_bank import Memory

# --- Setup ---
# Make sure to set your OPENAI_API_KEY environment variable
# export OPENAI_API_KEY='sk-...'

try:
    client = openai.OpenAI()
except openai.OpenAIError:
    print("Please set your OPENAI_API_KEY environment variable.")
    exit()

# 1. Instantiate the Memory Bank
# This will create a 'my_chatbot_memories' folder in your current directory
memory = Memory(storage_path="./my_chatbot_memories", user_id="user_jane_doe")

# 2. Wrap your existing OpenAI client
# This is the magic! The `chat_with_memory` object now has memory.
chat_with_memory = memory.wrap(client)

# --- Conversation ---
print("Chatbot with memory is ready. Type 'exit' to end.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    
    # 3. Use the wrapped client just like you would the original
    response = chat_with_memory.chat(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": user_input}]
    )
    
    print(f"Bot: {response}")
    if chat_with_memory.last_trace:
        print("\n--- Memory Trace ---")
        print(chat_with_memory.last_trace.summary())
        print("--------------------\n")