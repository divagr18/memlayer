"""
Example demonstrating the new direct knowledge ingestion APIs.

This example shows:
1. update_from_text() - Directly ingesting knowledge from text
2. synthesize_answer() - Querying with memory-grounded responses
"""

import os
from dotenv import load_dotenv
from memlayer.wrappers.openai import OpenAI

# Load environment variables
load_dotenv()

def main():
    print("=== Direct Knowledge Ingestion Example ===\n")
    
    # Initialize the memory-enhanced client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        storage_path="./direct_ingestion_demo",
        user_id="demo_user",
        operation_mode="online"
    )
    
    # --- Part 1: Direct Knowledge Ingestion ---
    print("Part 1: Ingesting knowledge from external documents\n")
    
    # Simulate ingesting knowledge from a document or email
    document_content = """
    Project Status Report - November 2025
    
    The Aurora Initiative is our flagship AI research project led by Dr. Sarah Chen.
    The project budget is $2.5 million, with completion expected by Q2 2026.
    
    Key milestones:
    - Phase 1 (Data Collection): Completed October 2025
    - Phase 2 (Model Training): In progress, 60% complete
    - Phase 3 (Deployment): Scheduled for March 2026
    
    The team consists of 8 researchers and 3 engineers. The project uses
    TensorFlow and PyTorch frameworks, hosted on AWS infrastructure.
    
    Recent achievement: The model achieved 94% accuracy on validation tests,
    exceeding the initial target of 90%.
    """
    
    print("Ingesting document content...")
    client.update_from_text(document_content)
    print("✓ Knowledge ingestion complete!\n")
    
    # Give the consolidation service a moment to process
    import time
    print("Waiting for consolidation to complete...")
    time.sleep(3)
    
    # --- Part 2: Memory-Grounded Question Answering ---
    print("\nPart 2: Querying with synthesize_answer()\n")
    
    questions = [
        "Who leads the Aurora Initiative?",
        "What is the budget for the Aurora Initiative?",
        "What frameworks are being used?",
        "What was the recent achievement?",
        "When is the project expected to be completed?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"{i}. Question: {question}")
        answer = client.synthesize_answer(question)
        print(f"   Answer: {answer}\n")
    
    # --- Part 3: Demonstrate return_object=True ---
    print("\nPart 3: Getting detailed answer object\n")
    
    question = "Tell me about the Aurora Initiative team composition"
    print(f"Question: {question}")
    
    answer_obj = client.synthesize_answer(question, return_object=True)
    print(f"\nAnswer: {answer_obj.answer}")
    print(f"\nContext used:\n{answer_obj.context[:200]}...")
    
    if answer_obj.trace:
        print(f"\nTrace information:")
        for event in answer_obj.trace.events:
            print(f"  - {event.name}: {event.duration_ms:.1f}ms")
    
    # Cleanup
    print("\n=== Example Complete ===")
    client.close()

if __name__ == "__main__":
    main()
