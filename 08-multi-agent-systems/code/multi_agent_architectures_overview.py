"""Multi-Agent Architectures Overview -- network-style peer routing between specialists.

Objective:
    Show a decentralized network of specialist agents (researcher -> writer ->
    reviewer) where each node appends its contribution and picks the next peer.
"""
import operator
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


# STEP 1 -- STATE: `notes` uses operator.add so every specialist appends without overwriting.
class NetworkState(TypedDict):
    task: str
    notes: Annotated[list[str], operator.add]
    next_agent: str


# STEP 2 -- SPECIALIST AGENT NODES.
def researcher(state: NetworkState) -> dict:
    # state coming IN:  {"task": "...", "notes": [], "next_agent": ""}
    # state going OUT:  appends research note, sets next_agent="writer"
    return {"notes": [f"[Researcher] Gathered facts for: {state['task']}"], "next_agent": "writer"}


def writer(state: NetworkState) -> dict:
    # state going OUT:  appends draft note, sets next_agent="reviewer"
    return {"notes": ["[Writer] Drafted summary from research notes."], "next_agent": "reviewer"}


def reviewer(state: NetworkState) -> dict:
    # state going OUT:  appends approval note, sets next_agent=END
    return {"notes": ["[Reviewer] Approved final summary."], "next_agent": END}


# STEP 3 -- ROUTING FUNCTION & GRAPH WIRING.
def route_peer(state: NetworkState) -> str:
    return state["next_agent"]


builder = StateGraph(NetworkState)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "researcher")
builder.add_conditional_edges("researcher", route_peer, {"writer": "writer", END: END})
builder.add_conditional_edges("writer", route_peer, {"reviewer": "reviewer", END: END})
builder.add_conditional_edges("reviewer", route_peer, {END: END})
graph = builder.compile()

# STEP 4 -- INVOKE: researcher -> writer -> reviewer -> END.
print("\n--- Multi-Agent Network (Peer Handoffs) ---")
result = graph.invoke({"task": "Explain LangGraph multi-agent patterns", "notes": [], "next_agent": ""})
for idx, line in enumerate(result["notes"], start=1):
    print(f"  Step {idx}: {line}")
