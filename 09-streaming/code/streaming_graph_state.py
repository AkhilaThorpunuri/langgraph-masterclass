"""Streaming Graph State -- stream() vs invoke() with values, updates, and debug modes.

Objective:
    Show how graph.stream() yields intermediate output after every node instead
    of waiting for the whole graph to finish, comparing updates, values, and debug.
"""
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
class StepState(TypedDict):
    text: str
    steps: list[str]


# STEP 2 -- NODES.
def clean(state: StepState) -> dict:
    # state coming IN:  {"text": "  LangGraph Streaming  ", "steps": []}
    # state going OUT:  {"text": "langgraph streaming", "steps": ["clean"]}
    return {"text": state["text"].strip().lower(), "steps": state["steps"] + ["clean"]}


def summarize(state: StepState) -> dict:
    # state coming IN:  {"text": "langgraph streaming", "steps": ["clean"]}
    # state going OUT:  {"text": "summary(langgraph streaming)", "steps": ["clean", "summarize"]}
    return {"text": f"summary({state['text']})", "steps": state["steps"] + ["summarize"]}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(StepState)
builder.add_node("clean", clean)
builder.add_node("summarize", summarize)
builder.add_edge(START, "clean")
builder.add_edge("clean", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

# STEP 4 -- STREAM IN ALL THREE MODES:
initial = {"text": "  LangGraph Streaming  ", "steps": []}

# updates: yields {node_name: partial_update} per node
print("\n--- Stream Mode: updates (per-node changes) ---")
for chunk in graph.stream(initial, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"  {node:>10} -> text={update['text']!r}, steps={update['steps']}")

# values: yields the full merged state after every superstep
print("\n--- Stream Mode: values (full state snapshots) ---")
for idx, chunk in enumerate(graph.stream(initial, stream_mode="values")):
    print(f"  Step {idx} -> text={chunk['text']!r}, steps={chunk['steps']}")

# debug: yields internal step/task execution events
print("\n--- Stream Mode: debug (internal events) ---")
for chunk in graph.stream(initial, stream_mode="debug"):
    print(f"  step={chunk['step']} | event={chunk['type']}")
