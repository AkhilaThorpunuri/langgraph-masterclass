"""Retry Policies and Error Handling -- RetryPolicy on nodes + fallback routing.

Objective:
    Show two layers of resilience:
    1) RetryPolicy auto-retries transient exceptions (e.g., 503 ConnectionError).
    2) Business-logic errors set state["error"] and route to a fallback_node.
"""
import socket
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


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

_ATTEMPTS = {"flaky_api": 0}


# STEP 1 -- STATE.
class ResilientState(TypedDict):
    query: str
    data: str
    error: str


# STEP 2 -- NODES.
def flaky_api(state: ResilientState) -> dict:
    # Attempt 1 raises ConnectionError -> RetryPolicy retries automatically.
    # Attempt 2 succeeds and returns live data.
    _ATTEMPTS["flaky_api"] += 1
    if _ATTEMPTS["flaky_api"] < 2:
        raise ConnectionError("Transient 503 from upstream API")
    return {"data": f"Fetched live data on attempt {_ATTEMPTS['flaky_api']}"}


def strict_parse(state: ResilientState) -> dict:
    # Records non-recoverable payload error in state instead of crashing.
    if "bad" in state["query"].lower():
        return {"error": "Malformed payload"}
    return {"error": ""}


def fallback_node(state: ResilientState) -> dict:
    return {"data": f"Fallback cached response (reason: {state['error']})"}


# STEP 3 -- ROUTING FUNCTION & GRAPH WIRING WITH RetryPolicy.
def route_on_error(state: ResilientState) -> str:
    return "fallback_node" if state.get("error") else END


builder = StateGraph(ResilientState)
builder.add_node(
    "flaky_api",
    flaky_api,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05),
)
builder.add_node("strict_parse", strict_parse)
builder.add_node("fallback_node", fallback_node)

builder.add_edge(START, "flaky_api")
builder.add_edge("flaky_api", "strict_parse")
builder.add_conditional_edges("strict_parse", route_on_error, {"fallback_node": "fallback_node", END: END})
builder.add_edge("fallback_node", END)
graph = builder.compile()

# STEP 4 -- INVOKE BOTH RECOVERY PATHS:
print("\n--- RetryPolicy & Fallback Error Handling ---")
r1 = graph.invoke({"query": "good request", "data": "", "error": ""})
print(f"  1. Transient API error -> {r1['data']}")

r2 = graph.invoke({"query": "bad request", "data": "", "error": ""})
print(f"  2. Payload error       -> {r2['data']}")
