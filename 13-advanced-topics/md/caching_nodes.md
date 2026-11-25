# Caching Nodes
> Venkata Bhattaram (c) 2026

## Contents
* CachePolicy
* Cache keys and TTLs
* When caching helps vs hurts

## Overview
When a node performs an expensive, deterministic lookup—such as embedding a document, calling a slow third-party API, or running a heavy computation—LangGraph can cache its output automatically. Attach **`cache_policy=CachePolicy(ttl=60)`** to the node and compile the graph with **`cache=InMemoryCache()`** (or a Redis/SQLite cache backend). Repeated calls with the same input within the TTL return the cached result immediately without re-executing the node.

Avoid caching nodes with side effects (sending emails, writing to databases) or nodes that must reflect real-time state.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 13-advanced-topics/code/caching_nodes.py
```

```python
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

print("First call  (miss):", graph.invoke({"key": "alpha", "value": ""}))
print("Second call (hit): ", graph.invoke({"key": "alpha", "value": ""}))
print("Third call  (new): ", graph.invoke({"key": "beta", "value": ""}))
print("Actual executions:", CALL_COUNTER["expensive_lookup"])
```

## Conclusion
`CachePolicy` + `InMemoryCache` cuts latency and API cost on pure, repeatable nodes while keeping TTL expiration declarative right on the node registration.
