# Environment Setup
> Venkata Bhattaram (c) 2026

## Contents
* Virtual environments
* API keys and `.env` / `settings.config`
* Local MLflow Tracing UI (`http://localhost:5000/`)

## Overview
Before running the course examples, set up an isolated Python virtual environment and install all course dependencies from the root `requirements.txt`. Most examples in chapters `01`–`14` run **offline without paid API keys**, and when you connect live LLM providers (Groq, Ollama Cloud, OpenRouter, OpenAI, Anthropic, or Gemini), keep keys out of source control using a `.env` file or `settings.config`.

## Setup Steps
```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify your LangGraph install
python 01-getting-started/code/verify_installation.py

# 4. (Optional) Start the local MLflow UI for tracing
mlflow ui --host 127.0.0.1 --port 5000
```

## Code Example
```python
import os
from dotenv import load_dotenv

load_dotenv()

for key in ("GROQ_API_KEY", "OLLAMA_API_KEY", "OPENROUTER_API_KEY"):
    status = "configured" if os.environ.get(key) else "not set"
    print(f"{key}: {status}")
```

## Conclusion
With `.venv` active, `requirements.txt` installed, and `mlflow ui` running on port `5000`, you can run any chapter script in `01`–`14` and inspect its execution traces under the **`masterclass`** experiment.
