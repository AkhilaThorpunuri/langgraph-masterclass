"""Custom Reducers and Channels -- writing custom merge functions for parallel updates.

Objective:
    Show how custom two-argument reducers `(left, right) -> merged` let parallel
    branches deduplicate lists and merge dictionaries without overwriting each other.
"""
import socket
from typing import Annotated, TypedDict

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


# STEP 1 -- CUSTOM REDUCERS: called by LangGraph whenever updates are merged into state.
def merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen = set(left)
    return left + [item for item in right if item not in seen]


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


# STEP 2 -- STATE: attach custom reducers via Annotated[T, reducer_fn].
class CustomReducerState(TypedDict):
    tags: Annotated[list[str], merge_unique]
    metrics: Annotated[dict[str, float], merge_dicts]


# STEP 3 -- PARALLEL BRANCH NODES (both run in the same superstep off START).
def branch_a(state: CustomReducerState) -> dict:
    return {"tags": ["python", "langgraph"], "metrics": {"latency_a_ms": 12.4}}


def branch_b(state: CustomReducerState) -> dict:
    # "langgraph" overlaps with branch_a -> merge_unique keeps it only once!
    return {"tags": ["langgraph", "mlflow"], "metrics": {"latency_b_ms": 18.7}}


# STEP 4 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(CustomReducerState)
builder.add_node("branch_a", branch_a)
builder.add_node("branch_b", branch_b)
builder.add_edge(START, "branch_a")
builder.add_edge(START, "branch_b")
builder.add_edge("branch_a", END)
builder.add_edge("branch_b", END)
graph = builder.compile()

# STEP 5 -- INVOKE:
# state coming IN:  {"tags": [], "metrics": {}}
# state going OUT:  {"tags": ["python", "langgraph", "mlflow"], "metrics": {"latency_a_ms": 12.4, "latency_b_ms": 18.7}}
print("\n--- Custom Reducers Demo (Parallel Branches) ---")
res = graph.invoke({"tags": [], "metrics": {}})
print(f"  Deduplicated tags: {res['tags']}")
print(f"  Merged metrics:    {res['metrics']}")
