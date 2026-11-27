"""Scaling and Production Configuration -- batching, concurrency limits, and recursion caps.

Objective:
    Show how to process multiple requests concurrently with graph.batch()
    while bounding concurrency (`max_concurrency`) and loop depth (`recursion_limit`).
"""
import socket
import time
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
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
class WorkState(TypedDict):
    item_id: int
    result: str


# STEP 2 -- NODES.
def process_item(state: WorkState) -> dict:
    # state coming IN:  {"item_id": N, "result": ""}
    # state going OUT:  {"result": "processed-item-N"}
    return {"result": f"processed-item-{state['item_id']}"}


# STEP 3 -- COMPILE WITH CHECKPOINTER (swap MemorySaver -> PostgresSaver in multi-worker prod).
builder = StateGraph(WorkState)
builder.add_node("process_item", process_item)
builder.add_edge(START, "process_item")
builder.add_edge("process_item", END)
graph = builder.compile(checkpointer=MemorySaver())

# STEP 4 -- BATCH EXECUTE WITH PER-REQUEST THREADS & BOUNDED CONCURRENCY:
inputs = [{"item_id": i, "result": ""} for i in range(1, 6)]
configs = [
    {"configurable": {"thread_id": f"prod-req-{i}"}, "recursion_limit": 10}
    for i in range(1, 6)
]

t0 = time.perf_counter()
results = graph.batch(inputs, config=configs, max_concurrency=2)
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"\n--- Batch Execution ({len(results)} requests, max_concurrency=2, {elapsed_ms:.1f} ms) ---")
for r in results:
    print(f"  Item {r['item_id']} -> {r['result']}")
