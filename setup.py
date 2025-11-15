from setuptools import setup, find_packages

setup(
    name="memory_bank",
    version="0.1.0",
    description="A lightweight memory layer for LLM applications",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.0.0",
    ],
    python_requires=">=3.8",
)
