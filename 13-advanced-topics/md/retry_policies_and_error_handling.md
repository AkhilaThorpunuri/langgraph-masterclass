# Retry Policies and Error Handling
> Venkata Bhattaram (c) 2026

## Contents
* RetryPolicy on nodes
* Catching and routing on errors
* Fallback nodes

## Overview
Production agents call flaky network APIs and receive malformed inputs. LangGraph handles both failure classes cleanly:
* **Transient failures** (timeouts, 503s): attach `retry_policy=RetryPolicy(max_attempts=3, initial_interval=...)` to `builder.add_node(...)` so LangGraph automatically retries with backoff.
* **Non-recoverable failures** (validation errors, missing records): record an `error` field in state and route via a conditional edge to a dedicated `fallback_node` that returns a safe degraded response.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 13-advanced-topics/code/retry_policies_and_error_handling.py
```

```python
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()

_ATTEMPTS = {"flaky_api": 0}


class ResilientState(TypedDict):
    query: str
    data: str
    error: str


def flaky_api(state: ResilientState) -> dict:
    _ATTEMPTS["flaky_api"] += 1
    if _ATTEMPTS["flaky_api"] < 2:
        raise ConnectionError("Transient 503 from upstream API")
    return {"data": f"Fetched live data on attempt {_ATTEMPTS['flaky_api']}"}


def strict_parse(state: ResilientState) -> dict:
    return {"error": "Malformed payload" if "bad" in state["query"].lower() else ""}


def fallback_node(state: ResilientState) -> dict:
    return {"data": f"Fallback cached response (reason: {state['error']})"}


builder = StateGraph(ResilientState)
builder.add_node("flaky_api", flaky_api, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05))
builder.add_node("strict_parse", strict_parse)
builder.add_node("fallback_node", fallback_node)
builder.add_edge(START, "flaky_api")
builder.add_edge("flaky_api", "strict_parse")
builder.add_conditional_edges("strict_parse", lambda s: "fallback_node" if s.get("error") else END, {"fallback_node": "fallback_node", END: END})
builder.add_edge("fallback_node", END)
graph = builder.compile()

print(graph.invoke({"query": "good request", "data": "", "error": ""}))
print(graph.invoke({"query": "bad request", "data": "", "error": ""}))
```

## Conclusion
Use `RetryPolicy` for transient infrastructure glitches and explicit fallback nodes for business-logic errors so your graphs degrade gracefully instead of crashing.
