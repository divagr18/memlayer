import time
import os
import shutil
from memory_bank.wrappers.openai import OpenAI

# --- Configuration ---
# Use a separate storage path for this example to keep memories isolated.
STORAGE_PATH = "./proactive_test_memory"
USER_ID = "proactive_user_001"

# --- Helper Functions ---
def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def cleanup():
    """Removes the storage directory created during the test."""
    print_header("Cleaning up")
    if os.path.exists(STORAGE_PATH):
        shutil.rmtree(STORAGE_PATH)
        print(f"Removed storage directory: {STORAGE_PATH}")

# --- Main Test Execution ---
def run_example():
    """
    This example demonstrates the full lifecycle of a proactive task:
    1. The user asks the LLM to schedule a reminder for a few seconds in the future.
    2. The LLM uses the `schedule_task` tool to save this to the knowledge graph.
    3. We wait for the task to become due.
    4. The user starts a new conversation.
    5. The LLM, prompted by the `SchedulerService`, proactively reminds the user.
    """
    # Ensure the OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable to run this example.")
        return

    # Clean up any previous runs before starting
    cleanup()

    print_header("Step 1: Initialize the Memory-Enhanced Client")
    
    # We set a very short scheduler interval for this demo to see results quickly.
    # In a real application, this would be 60 seconds or more.
    client = OpenAI(
        storage_path=STORAGE_PATH,
        user_id=USER_ID,
        scheduler_interval_seconds=5 # Check for due tasks every 5 seconds
    )
    
    print(f"Client initialized for user '{USER_ID}'.")
    print("Scheduler is running in the background, checking for tasks every 5 seconds.")

    # --------------------------------------------------------------------------
    
    print_header("Step 2: Schedule a Task")
    print("User: 'Hey, can you remind me in 15 seconds to check the oven?'")
    
    # The LLM will see this and should decide to call the `schedule_task` tool.
    # The due date will be calculated by the LLM based on the current time.
    response = client.chat(messages=[
        {"role": "user", "content": "Hey, can you remind me in 15 seconds to check the oven?"}
    ])
    
    print(f"\nBot: {response}")
    print("\nBehind the scenes, the LLM called the `schedule_task` tool and a 'Task' node was created in the knowledge graph.")

    # --------------------------------------------------------------------------

    print_header("Step 3: Wait for the Task to Become Due")
    wait_time = 20
    print(f"Waiting for {wait_time} seconds. During this time, the background scheduler will run, find the due task, and change its status to 'triggered'.")
    
    for i in range(wait_time, 0, -1):
        print(f"...{i}", end="", flush=True)
        time.sleep(1)
    print("\nWait complete. The task should now be triggered.")

    # --------------------------------------------------------------------------

    print_header("Step 4: Start a New Conversation to Receive the Reminder")
    print("Now, we'll start a completely unrelated conversation.")
    print("The memory system will proactively inject the reminder into the prompt.")
    print("\nUser: 'What's the weather like today?'")

    # The user's message is simple, but the system will prepend the reminder context.
    # The LLM should both answer the question AND deliver the reminder.
    response = client.chat(messages=[
        {"role": "user", "content": "What's the weather like today?"}
    ])

    print(f"\nBot: {response}")
    print("\nNotice how the bot both answered the question and delivered the reminder.")
    print("The task's status in the knowledge graph has now been set to 'completed' to prevent it from being sent again.")

    # --------------------------------------------------------------------------

    # Gracefully shut down the background scheduler thread
    client.close()
    cleanup()
    print_header("Example Complete")

if __name__ == "__main__":
    run_example()