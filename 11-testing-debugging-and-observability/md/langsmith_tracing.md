# LangSmith & MLflow Tracing
> Venkata Bhattaram (c) 2026

## Contents
* Enabling tracing (MLflow locally & LangSmith in the cloud)
* Inspecting runs, tags, and metadata
* Comparing traces across executions

## Overview
Observability is essential once your graph has conditional branches, loops, tools, or multiple agents. LangGraph integrates seamlessly with both:
* **Local MLflow Tracing**: set `mlflow.set_tracking_uri("http://localhost:5000/")`, `mlflow.set_experiment("masterclass")`, and `mlflow.langchain.autolog()` to capture every node span locally with zero cloud account needed.
* **LangSmith Cloud Tracing**: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` in your environment to mirror runs to LangSmith.

Passing `run_name`, `tags`, and `metadata` inside `config={...}` makes traces easy to filter and group in either UI.

## Code Example
Run the standalone script (logs to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 11-testing-debugging-and-observability/code/langsmith_tracing.py
```

```python
import os
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class TraceState(TypedDict):
    text: str
    word_count: int
    summary: str


def count_words(state: TraceState) -> dict:
    return {"word_count": len(state["text"].split())}


def summarize(state: TraceState) -> dict:
    return {"summary": f"Input has {state['word_count']} words."}


builder = StateGraph(TraceState)
builder.add_node("count_words", count_words)
builder.add_node("summarize", summarize)
builder.add_edge(START, "count_words")
builder.add_edge("count_words", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

result = graph.invoke(
    {"text": "Observability makes complex LangGraph agents easy to debug", "word_count": 0, "summary": ""},
    config={"run_name": "observability-demo-run", "tags": ["observability", "masterclass"], "metadata": {"env": "local"}},
)
print("Result:", result)
```

## Conclusion
Always attach meaningful `run_name`, `tags`, and `metadata` to `invoke()` calls—they turn raw trace logs into searchable diagnostics when debugging production runs.
