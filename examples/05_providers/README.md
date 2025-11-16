# Provider Examples

This folder contains examples for each supported LLM provider.

## 🎯 Quick Start

Choose your provider and run the corresponding example:

### OpenAI (GPT models)
```bash
export OPENAI_API_KEY='sk-...'
python examples/05_providers/openai_example.py
```
**Models**: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo

### Claude (Anthropic)
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
python examples/05_providers/claude_example.py
```
**Models**: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus

### Google Gemini
```bash
export GOOGLE_API_KEY='...'
python examples/05_providers/gemini_example.py
```
**Models**: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash

### Ollama (Local)
```bash
# No API key needed! Run locally
ollama pull qwen3:1.7b
python examples/05_providers/ollama_example.py
```
**Models**: llama3.2, qwen3, mistral, phi3, deepseek-r1, and 100+ more

## 🔑 API Key Setup

### Linux/Mac
```bash
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'
export GOOGLE_API_KEY='your-key-here'
```

### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY='your-key-here'
$env:ANTHROPIC_API_KEY='your-key-here'
$env:GOOGLE_API_KEY='your-key-here'
```

### Windows (CMD)
```cmd
set OPENAI_API_KEY=your-key-here
set ANTHROPIC_API_KEY=your-key-here
set GOOGLE_API_KEY=your-key-here
```

## 📊 Provider Comparison

| Provider | Best For | Cost | Speed | Quality |
|----------|----------|------|-------|---------|
| **OpenAI** | General purpose, production apps | $$$ | Fast | Excellent |
| **Claude** | Long context, detailed analysis | $$$ | Medium | Excellent |
| **Gemini** | Multimodal, fast prototyping | $$ | Very Fast | Good |
| **Ollama** | Privacy, offline, no cost | FREE | Variable | Good |

## 🎨 Common Usage Pattern

All providers share the same API:

```python
from memlayer.wrappers.{provider} import {Provider}

# Initialize with memory
client = Provider(
    model="model-name",
    storage_path="./memories",
    user_id="user_id",
    salience_mode="local"  # or "online" or "lightweight"
)

# Chat with automatic memory
response = client.chat(messages=[
    {"role": "user", "content": "Remember my name is Alice"}
])

# Memory is automatically stored and retrieved!
```

## 🚀 Salience Modes

Each provider supports three memory modes:

### 1. **Local Mode** (Default)
- Uses sentence-transformers locally
- High accuracy semantic filtering
- ~10s startup time, no API costs
- Best for: High-volume usage, offline apps

### 2. **Online Mode**
- Uses OpenAI embeddings API
- Fast startup (~2s), small API cost
- Requires OPENAI_API_KEY
- Best for: Production apps, serverless

### 3. **Lightweight Mode**
- Keyword-based filtering only
- Instant startup, no embeddings
- Best for: Prototyping, testing

Example:
```python
client = Provider(
    model="...",
    salience_mode="online",  # Fast startup!
    storage_path="./memories"
)
```

## 💡 Tips

1. **Mix and match**: Use different providers for different tasks
2. **Ollama for dev**: Use Ollama locally, then switch to cloud provider for production
3. **Online mode for serverless**: Use `salience_mode="online"` in AWS Lambda/Cloud Functions
4. **Share memories**: All providers can read the same memory storage if using same `storage_path`

## 📚 More Examples

- **Basics**: `examples/01_basics/`
- **Search Tiers**: `examples/02_search_tiers/`
- **Advanced Features**: `examples/03_features/`
- **Benchmarks**: `examples/04_benchmarks/`
