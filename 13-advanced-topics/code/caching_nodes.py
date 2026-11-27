"""Caching Nodes -- CachePolicy and InMemoryCache to avoid recomputing expensive nodes.

Objective:
    Show how CachePolicy(ttl=60) + InMemoryCache() serves repeat calls with the
    same input directly from cache without re-running the node function.
"""
import socket
from typing import TypedDict

import mlflow
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy


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

CALL_COUNTER = {"expensive_lookup": 0}


# STEP 1 -- STATE.
class CacheState(TypedDict):
    key: str
    value: str


# STEP 2 -- EXPENSIVE PURE NODE (only runs on cache misses).
def expensive_lookup(state: CacheState) -> dict:
    # Increments CALL_COUNTER only when actually executed (not served from cache).
    CALL_COUNTER["expensive_lookup"] += 1
    return {"value": f"computed({state['key']}) [exec #{CALL_COUNTER['expensive_lookup']}]"}


# STEP 3 -- GRAPH & COMPILE WITH CachePolicy + InMemoryCache.
builder = StateGraph(CacheState)
builder.add_node("expensive_lookup", expensive_lookup, cache_policy=CachePolicy(ttl=60))
builder.add_edge(START, "expensive_lookup")
builder.add_edge("expensive_lookup", END)
graph = builder.compile(cache=InMemoryCache())

# STEP 4 -- INVOKE 3 TIMES (2 unique keys -> only 2 actual node executions):
print("\n--- Node Caching Demo (CachePolicy ttl=60s) ---")
c1 = graph.invoke({"key": "alpha", "value": ""})
c2 = graph.invoke({"key": "alpha", "value": ""})
c3 = graph.invoke({"key": "beta", "value": ""})
print(f"  Call 1 (alpha - miss) -> {c1['value']}")
print(f"  Call 2 (alpha - hit)  -> {c2['value']}")
print(f"  Call 3 (beta  - miss) -> {c3['value']}")
print(f"  Actual node runs:        {CALL_COUNTER['expensive_lookup']} (saved 1 execution!)")
