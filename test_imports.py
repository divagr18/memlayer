"""
Test script to verify that all wrapper imports work and are fast
"""
import time

print("Testing import speeds...\n")

# Test OpenAI import
start = time.time()
from memory_bank.wrappers.openai import OpenAI
elapsed = time.time() - start
print(f"✓ OpenAI import: {elapsed:.3f}s")

# Test Claude import
start = time.time()
from memory_bank.wrappers.claude import Claude
elapsed = time.time() - start
print(f"✓ Claude import: {elapsed:.3f}s")

# Test Gemini import
start = time.time()
from memory_bank.wrappers.gemini import Gemini
elapsed = time.time() - start
print(f"✓ Gemini import: {elapsed:.3f}s")

# Test Ollama import
start = time.time()
from memory_bank.wrappers.ollama import Ollama
elapsed = time.time() - start
print(f"✓ Ollama import: {elapsed:.3f}s")

# Test package-level imports
start = time.time()
from memory_bank import OpenAI, Claude, Gemini, Ollama
elapsed = time.time() - start
print(f"\n✓ All package-level imports: {elapsed:.3f}s")

print("\n✅ All imports successful!")
print("\nClasses available:")
print(f"  - OpenAI: {OpenAI}")
print(f"  - Claude: {Claude}")
print(f"  - Gemini: {Gemini}")
print(f"  - Ollama: {Ollama}")
