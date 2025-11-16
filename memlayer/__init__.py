"""
MemLayer - A lightweight memory layer for LLM applications
"""

__version__ = "0.1.0"

# Lazy imports to speed up package loading
def __getattr__(name):
    if name == "Memory":
        from .client import Memory
        return Memory
    elif name == "OpenAI":
        from .wrappers import OpenAI
        return OpenAI
    elif name == "Claude":
        from .wrappers import Claude
        return Claude
    elif name == "Gemini":
        from .wrappers import Gemini
        return Gemini
    elif name == "Ollama":
        from .wrappers import Ollama
        return Ollama
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["Memory", "OpenAI", "Claude", "Gemini", "Ollama"]

