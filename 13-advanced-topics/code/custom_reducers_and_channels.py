"""Custom Reducers and Channels -- writing custom merge functions for parallel updates.

Demonstrates:
- Custom deduplicating list reducer (keeps order, drops duplicates)
- Custom dict-merge reducer for combining partial metadata from parallel branches
"""
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


def merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen = set(left)
    return left + [item for item in right if item not in seen]


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class CustomReducerState(TypedDict):
    tags: Annotated[list[str], merge_unique]
    metrics: Annotated[dict[str, float], merge_dicts]


def branch_a(state: CustomReducerState) -> dict:
    return {"tags": ["python", "langgraph"], "metrics": {"latency_a_ms": 12.4}}


def branch_b(state: CustomReducerState) -> dict:
    return {"tags": ["langgraph", "mlflow"], "metrics": {"latency_b_ms": 18.7}}


builder = StateGraph(CustomReducerState)
builder.add_node("branch_a", branch_a)
builder.add_node("branch_b", branch_b)
builder.add_edge(START, "branch_a")
builder.add_edge(START, "branch_b")
builder.add_edge("branch_a", END)
builder.add_edge("branch_b", END)
graph = builder.compile()

res = graph.invoke({"tags": [], "metrics": {}})
print("Deduplicated tags:", res["tags"])
print("Merged metrics:   ", res["metrics"])
