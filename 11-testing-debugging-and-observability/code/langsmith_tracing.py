"""Tracing & Observability -- LangSmith env config + local MLflow Tracing.

Objective:
    Show how to attach run_name, tags, and metadata to graph.invoke() so every
    run is easy to filter in MLflow (local) or LangSmith (cloud).
"""
import os
import socket
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if _is_port_open("127.0.0.1", 5000):
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("masterclass")
    mlflow.langchain.autolog()
else:
    print("[MLflow] Server not running at http://localhost:5000/ -- skipping remote tracing.")


# STEP 1 -- STATE.
class TraceState(TypedDict):
    text: str
    word_count: int
    summary: str


# STEP 2 -- NODES.
def count_words(state: TraceState) -> dict:
    # state coming IN:  {"text": "...", "word_count": 0, "summary": ""}
    # state going OUT:  {"word_count": 9}
    return {"word_count": len(state["text"].split())}


def summarize(state: TraceState) -> dict:
    # state coming IN:  {"word_count": 9, ...}
    # state going OUT:  {"summary": "Input has 9 words."}
    return {"summary": f"Input has {state['word_count']} words."}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(TraceState)
builder.add_node("count_words", count_words)
builder.add_node("summarize", summarize)
builder.add_edge(START, "count_words")
builder.add_edge("count_words", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

# STEP 4 -- INVOKE WITH TRACE METADATA (run_name, tags, metadata):
print("\n--- Tracing & Observability Demo ---")
langsmith_enabled = bool(os.environ.get("LANGSMITH_API_KEY") and os.environ.get("LANGSMITH_TRACING") == "true")
print(f"  LangSmith cloud tracing: {'ON' if langsmith_enabled else 'OFF (local MLflow active)'}")

result = graph.invoke(
    {"text": "Observability makes complex LangGraph agents easy to debug", "word_count": 0, "summary": ""},
    config={
        "run_name": "observability-demo-run",
        "tags": ["observability", "masterclass"],
        "metadata": {"env": "local"},
    },
)
print(f"  Input:      {result['text']}")
print(f"  Word count: {result['word_count']}")
print(f"  Summary:    {result['summary']}")
