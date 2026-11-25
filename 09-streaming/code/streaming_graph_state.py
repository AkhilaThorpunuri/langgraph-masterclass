"""Streaming Graph State -- stream() vs invoke() with values, updates, and debug modes.

Demonstrates:
- Streaming full state snapshots with stream_mode="values"
- Streaming per-node diffs with stream_mode="updates"
- Streaming detailed execution metadata with stream_mode="debug"
"""
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class StepState(TypedDict):
    text: str
    steps: list[str]


def clean(state: StepState) -> dict:
    return {"text": state["text"].strip().lower(), "steps": state["steps"] + ["clean"]}


def summarize(state: StepState) -> dict:
    return {"text": f"summary({state['text']})", "steps": state["steps"] + ["summarize"]}


builder = StateGraph(StepState)
builder.add_node("clean", clean)
builder.add_node("summarize", summarize)
builder.add_edge(START, "clean")
builder.add_edge("clean", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

initial = {"text": "  LangGraph Streaming  ", "steps": []}

print("=== stream_mode='updates' ===")
for chunk in graph.stream(initial, stream_mode="updates"):
    print(" ", chunk)

print("\n=== stream_mode='values' ===")
for chunk in graph.stream(initial, stream_mode="values"):
    print(" ", chunk)

print("\n=== stream_mode='debug' ===")
for chunk in graph.stream(initial, stream_mode="debug"):
    print(" ", chunk["type"], "step=", chunk["step"])
