"""Retry Policies and Error Handling -- RetryPolicy on nodes + fallback routing.

Demonstrates:
- Attaching RetryPolicy(max_attempts=3) to a flaky node so transient errors auto-retry
- Catching non-recoverable errors in state and routing to a dedicated fallback node
"""
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
    if "bad" in state["query"].lower():
        return {"error": "Malformed payload"}
    return {"error": ""}


def fallback_node(state: ResilientState) -> dict:
    return {"data": f"Fallback cached response (reason: {state['error']})"}


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

print("1. Transient error auto-recovered via RetryPolicy:")
print(" ", graph.invoke({"query": "good request", "data": "", "error": ""}))

print("2. Unrecoverable error routed to fallback_node:")
print(" ", graph.invoke({"query": "bad request", "data": "", "error": ""}))
