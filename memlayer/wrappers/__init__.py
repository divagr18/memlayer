"""
LLM client wrappers for Memlayer
"""


# Lazy imports for fast package loading
def __getattr__(name):
    if name == "OpenAI":
        from .openai import OpenAI

        return OpenAI
    elif name == "Claude":
        from .claude import Claude

        return Claude
    elif name == "Novita":
        from .novita import Novita

        return Novita
    elif name == "LMStudio":
        from .lmstudio import LMStudio

        return LMStudio
    elif name == "Gemini":
        from .gemini import Gemini

        return Gemini
    elif name == "Ollama":
        from .ollama import Ollama

        return Ollama
    elif name == "OpenAIWrapper":
        from .openai import OpenAIWrapper

        return OpenAIWrapper
    elif name == "ClaudeWrapper":
        from .claude import ClaudeWrapper

        return ClaudeWrapper
    elif name == "NovitaWrapper":
        from .novita import Novita

        return Novita
    elif name == "GeminiWrapper":
        from .gemini import GeminiWrapper

        return GeminiWrapper
    elif name == "OllamaWrapper":
        from .ollama import OllamaWrapper

        return OllamaWrapper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "OpenAI",  # New standalone class
    "Claude",  # New standalone class
    "Novita",  # New standalone class (Novita.ai provider)
    "Gemini",  # New standalone class
    "Ollama",
    "LMStudio",  # New standalone class
    "OpenAIWrapper",  # Legacy wrapper
    "ClaudeWrapper",  # Legacy wrapper
    "NovitaWrapper",  # Legacy wrapper (alias for Novita)
    "GeminiWrapper",  # Legacy wrapper
    "OllamaWrapper",  # Legacy wrapper
]
