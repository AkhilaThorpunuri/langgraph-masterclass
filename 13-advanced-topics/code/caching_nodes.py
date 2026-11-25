"""Caching Nodes -- CachePolicy and InMemoryCache to avoid recomputing expensive nodes.

Demonstrates:
- Attaching CachePolicy(ttl=60) to an expensive node
- Compiling with InMemoryCache() and proving the second call with the same input hits cache
"""
from typing import TypedDict

import mlflow
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()

CALL_COUNTER = {"expensive_lookup": 0}


class CacheState(TypedDict):
    key: str
    value: str


def expensive_lookup(state: CacheState) -> dict:
    CALL_COUNTER["expensive_lookup"] += 1
    return {"value": f"computed({state['key']}) [exec #{CALL_COUNTER['expensive_lookup']}]"}


builder = StateGraph(CacheState)
builder.add_node("expensive_lookup", expensive_lookup, cache_policy=CachePolicy(ttl=60))
builder.add_edge(START, "expensive_lookup")
builder.add_edge("expensive_lookup", END)

graph = builder.compile(cache=InMemoryCache())

print("First call  (cache miss):", graph.invoke({"key": "alpha", "value": ""}))
print("Second call (cache hit): ", graph.invoke({"key": "alpha", "value": ""}))
print("Third call  (new key):   ", graph.invoke({"key": "beta", "value": ""}))
print("Total actual executions of expensive_lookup:", CALL_COUNTER["expensive_lookup"])
