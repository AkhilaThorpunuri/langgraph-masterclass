"""Graph Visualization -- Mermaid diagram export and topology inspection.

Objective:
    Show how graph.get_graph().draw_mermaid() prints a ready-to-paste Mermaid
    flowchart of any compiled graph so you can visually verify edges & branches.
"""
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
class VizState(TypedDict):
    score: int
    status: str


# STEP 2 -- NODES & ROUTING FUNCTION.
def evaluate(state: VizState) -> dict:
    return {}


def approve(state: VizState) -> dict:
    return {"status": "approved"}


def reject(state: VizState) -> dict:
    return {"status": "rejected"}


def route(state: VizState) -> str:
    return "approve" if state["score"] >= 70 else "reject"


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(VizState)
builder.add_node("evaluate", evaluate)
builder.add_node("approve", approve)
builder.add_node("reject", reject)
builder.add_edge(START, "evaluate")
builder.add_conditional_edges("evaluate", route, {"approve": "approve", "reject": "reject"})
builder.add_edge("approve", END)
builder.add_edge("reject", END)
graph = builder.compile()

# STEP 4 -- EXPORT MERMAID DIAGRAM & RUN SAMPLE INVOCATION:
print("\n--- Mermaid Graph Diagram ---")
print(graph.get_graph().draw_mermaid().strip())

out = graph.invoke({"score": 85, "status": ""})
print("\n--- Sample Run ---")
print(f"  Score {out['score']} -> {out['status']}")
