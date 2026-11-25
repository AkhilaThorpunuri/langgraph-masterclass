"""Tracing & Observability -- LangSmith env config + local MLflow Tracing.

Demonstrates:
- Attaching run_name, tags, and metadata to a graph.invoke() call
- Logging traces locally to MLflow (http://localhost:5000/, experiment 'masterclass')
- Showing the env vars that also enable LangSmith cloud tracing when LANGSMITH_API_KEY is set
"""
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

langsmith_enabled = bool(os.environ.get("LANGSMITH_API_KEY") and os.environ.get("LANGSMITH_TRACING") == "true")
print(f"LangSmith cloud tracing enabled: {langsmith_enabled} (MLflow local tracing is active)")

result = graph.invoke(
    {"text": "Observability makes complex LangGraph agents easy to debug", "word_count": 0, "summary": ""},
    config={
        "run_name": "observability-demo-run",
        "tags": ["observability", "masterclass"],
        "metadata": {"env": "local"},
    },
)
print("Result:", result)
