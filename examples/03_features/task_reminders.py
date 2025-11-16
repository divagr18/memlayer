import time
import os
import shutil
from memlayer.wrappers.openai import OpenAI

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
        try:
            shutil.rmtree(STORAGE_PATH)
            print(f"Removed storage directory: {STORAGE_PATH}")
        except PermissionError as e:
            print(f"Note: Could not remove {STORAGE_PATH} - files may be in use.")
            print(f"You can manually delete this directory later if needed.")
            print(f"Error details: {e}")

# --- Main Test Execution ---
def run_example():
    """
    This example demonstrates the full lifecycle of a proactive task reminder:
    1. User asks the LLM to schedule a reminder for a few seconds in the future.
    2. LLM uses the `schedule_task` tool to save it to the knowledge graph (status: 'pending').
    3. We wait for the task's due time to pass.
    4. User starts a new conversation.
    5. At the start of chat(), the system checks for due 'pending' tasks.
    6. LLM receives the reminder context and informs the user.
    7. Task status is updated to 'completed' to prevent re-showing.
    """
    # Ensure the OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable to run this example.")
        return

    # Clean up any previous runs before starting
    cleanup()

    print_header("Step 1: Initialize the Memory-Enhanced Client")
    
    # The memory system checks for due tasks on every chat() call.
    # No background scheduler is needed - tasks are checked when the user interacts.
    client = OpenAI(
        storage_path=STORAGE_PATH,
        user_id=USER_ID
    )
    
    print(f"Client initialized for user '{USER_ID}'.")
    print("The system will check for due tasks at the start of each conversation turn.")

    # --------------------------------------------------------------------------
    
    print_header("Step 2: Schedule a Task")
    print("User: 'Hey, can you remind me in 15 seconds to check the oven?'")
    
    # The LLM will see this and should decide to call the `schedule_task` tool.
    # The due date will be calculated by the LLM based on the current time.
    response = client.chat(messages=[
        {"role": "user", "content": "Hey, can you remind me in 15 seconds to check the oven?"}
    ])
    
    print(f"\nBot: {response}")
    print("\nBehind the scenes, the LLM called the `schedule_task` tool and created a Task node in the knowledge graph.")
    print("The task is now in 'pending' status with a due_timestamp 15 seconds from now.")

    # --------------------------------------------------------------------------

    print_header("Step 3: Wait for the Task to Become Due")
    wait_time = 20
    print(f"Waiting for {wait_time} seconds to ensure the task's due time has passed.")
    print("When we start a new conversation, the system will automatically check for due tasks.")
    
    for i in range(wait_time, 0, -1):
        print(f"...{i}", end="", flush=True)
        time.sleep(1)
    print("\nWait complete. The task should now be past its due time.")

    # --------------------------------------------------------------------------

    print_header("Step 4: Start a New Conversation to Receive the Reminder")
    print("Now, we'll start a completely unrelated conversation.")
    print("The memory system will automatically check for due tasks and inject reminders.")
    print("\nUser: 'What's the weather like today?'")

    # At the start of this chat() call, get_triggered_tasks_context() will be called.
    # It will find pending tasks where due_timestamp <= current_time and inject them.
    # The LLM should both answer the question AND deliver the reminder.
    response = client.chat(messages=[
        {"role": "user", "content": "What's the weather like today?"}
    ])

    print(f"\nBot: {response}")
    print("\nNotice how the bot both answered the question and delivered the reminder!")
    print("The task status has been updated to 'completed' to prevent it from being shown again.")

    # --------------------------------------------------------------------------

    # Clean up references to allow ChromaDB to close properly
    print("\nReleasing storage resources...")
    client.close()
    time.sleep(0.5)  # Give ChromaDB time to release file locks
    
    cleanup()
    print_header("Example Complete")

if __name__ == "__main__":
    run_example()