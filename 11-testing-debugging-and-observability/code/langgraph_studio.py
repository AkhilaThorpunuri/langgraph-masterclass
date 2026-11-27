"""LangGraph Studio -- stepping through execution and editing state mid-run.

Objective:
    Show the exact checkpoint APIs that power LangGraph Studio: pause before a
    node with interrupt_before, inspect state, edit it with update_state, and resume.
"""
import socket
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
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
class StudioState(TypedDict):
    query: str
    plan: str
    result: str


# STEP 2 -- NODES.
def plan_node(state: StudioState) -> dict:
    # state coming IN:  {"query": "observability", "plan": "", "result": ""}
    # state going OUT:  {"plan": "Step 1: research observability"}
    return {"plan": f"Step 1: research {state['query']}"}


def execute_node(state: StudioState) -> dict:
    # state coming IN:  {"plan": "<possibly edited in Studio>", ...}
    # state going OUT:  {"result": "Executed -> ..."}
    return {"result": f"Executed -> {state['plan']}"}


# STEP 3 -- GRAPH & COMPILE WITH BREAKPOINT:
# Exported as module-level `graph` so `langgraph dev` / Studio can load it directly.
builder = StateGraph(StudioState)
builder.add_node("plan_node", plan_node)
builder.add_node("execute_node", execute_node)
builder.add_edge(START, "plan_node")
builder.add_edge("plan_node", "execute_node")
builder.add_edge("execute_node", END)

graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["execute_node"])

# STEP 4 -- SIMULATE STUDIO STEP-THROUGH & MID-RUN STATE EDIT:
if __name__ == "__main__":
    print("\n--- LangGraph Studio Step-Through Demo ---")
    cfg = {"configurable": {"thread_id": "studio-step-1"}}
    graph.invoke({"query": "observability", "plan": "", "result": ""}, cfg)
    snap = graph.get_state(cfg)
    print(f"  1. Paused before: {snap.next[0]}")
    print(f"     Original plan: {snap.values['plan']}")

    graph.update_state(cfg, {"plan": "Step 1: research observability + MLflow tracing"})
    final = graph.invoke(None, cfg)
    print(f"  2. Edited plan:   {final['plan']}")
    print(f"  3. Final result:  {final['result']}")
