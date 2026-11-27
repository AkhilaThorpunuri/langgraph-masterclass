"""LangGraph Platform and Server -- structuring a graph export and langgraph.json config.

Objective:
    Show the module-level `graph` export contract expected by `langgraph dev`
    and LangGraph Platform's `langgraph.json` manifest.
"""
import json
import socket
from typing import TypedDict

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


# STEP 1 -- STATE.
class EchoState(TypedDict):
    message: str
    reply: str


# STEP 2 -- NODES.
def respond(state: EchoState) -> dict:
    # state coming IN:  {"message": "Hello Platform!", "reply": ""}
    # state going OUT:  {"reply": "LangGraph Server received: Hello Platform!"}
    return {"reply": f"LangGraph Server received: {state['message']}"}


# STEP 3 -- COMPILE & EXPORT `graph` AT MODULE LEVEL (loaded via langgraph.json).
builder = StateGraph(EchoState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile()

# STEP 4 -- MANIFEST EXAMPLE & LOCAL SMOKE TEST.
LANGGRAPH_JSON_EXAMPLE = {
    "dependencies": ["."],
    "graphs": {"echo": "./langgraph_platform_and_server.py:graph"},
    "env": ".env",
}

if __name__ == "__main__":
    print("\n--- LangGraph Platform Manifest (langgraph.json) ---")
    print(json.dumps(LANGGRAPH_JSON_EXAMPLE, indent=2))
    out = graph.invoke({"message": "Hello Platform!", "reply": ""})
    print("\n--- Local Invoke Test ---")
    print(f"  Input: {out['message']}")
    print(f"  Reply: {out['reply']}")
