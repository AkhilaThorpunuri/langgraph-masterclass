"""Scaling and Production Configuration -- batching, concurrency limits, and recursion caps.

Demonstrates:
- Running concurrent inputs with graph.batch(..., config={"max_concurrency": 2})
- Enforcing recursion_limit and timing/monitoring metrics per request
"""
import time
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class WorkState(TypedDict):
    item_id: int
    result: str


def process_item(state: WorkState) -> dict:
    return {"result": f"processed-item-{state['item_id']}"}


builder = StateGraph(WorkState)
builder.add_node("process_item", process_item)
builder.add_edge(START, "process_item")
builder.add_edge("process_item", END)

# Swap MemorySaver for PostgresSaver in multi-worker production deployments
graph = builder.compile(checkpointer=MemorySaver())

inputs = [{"item_id": i, "result": ""} for i in range(1, 6)]
configs = [
    {"configurable": {"thread_id": f"prod-req-{i}"}, "recursion_limit": 10}
    for i in range(1, 6)
]

t0 = time.perf_counter()
results = graph.batch(inputs, config=configs, max_concurrency=2)
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"Processed {len(results)} requests in {elapsed_ms:.1f} ms (max_concurrency=2):")
for r in results:
    print(" ", r)
