"""Interrupts -- static breakpoints (interrupt_before) and dynamic interrupt().

Objective:
    Show both ways to pause a graph for human-in-the-loop approval:
    1) Dynamic `interrupt(payload)` inside a node, resumed via Command(resume=...).
    2) Static `interrupt_before=["publish"]` at compile time, resumed via invoke(None).
"""
import socket
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


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
class ApprovalState(TypedDict):
    draft: str
    approved: bool
    status: str


# STEP 2 -- NODES.
def create_draft(state: ApprovalState) -> dict:
    # state coming IN:  {"draft": "", "approved": False, "status": ""}
    # state going OUT: {"draft": "Release v2.0 deployment notes"}
    return {"draft": "Release v2.0 deployment notes"}


def human_gate(state: ApprovalState) -> dict:
    # Pauses execution here and returns payload under result["__interrupt__"].
    # When resumed with Command(resume=True), `decision` receives True.
    decision = interrupt({"action": "approve_release", "draft": state["draft"]})
    return {"approved": bool(decision)}


def publish(state: ApprovalState) -> dict:
    # state coming IN:  {"approved": True/False, ...}
    # state going OUT: {"status": "published" | "cancelled"}
    return {"status": "published" if state["approved"] else "cancelled"}


# STEP 3 -- GRAPH, EDGES & COMPILE WITH STATIC BREAKPOINT.
builder = StateGraph(ApprovalState)
builder.add_node("create_draft", create_draft)
builder.add_node("human_gate", human_gate)
builder.add_node("publish", publish)
builder.add_edge(START, "create_draft")
builder.add_edge("create_draft", "human_gate")
builder.add_edge("human_gate", "publish")
builder.add_edge("publish", END)

graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["publish"])
config = {"configurable": {"thread_id": "hitl-interrupt-demo"}}

# STEP 4 -- RUN & RESUME THROUGH BOTH INTERRUPTS:
print("\n--- Interrupts Demo ---")
paused_1 = graph.invoke({"draft": "", "approved": False, "status": ""}, config)
prompt_1 = paused_1["__interrupt__"][0].value
print(f"  1. Dynamic interrupt -> Action: {prompt_1['action']} | Draft: {prompt_1['draft']!r}")

graph.invoke(Command(resume=True), config)
print(f"  2. Static breakpoint -> Approved: True | Paused before node: {graph.get_state(config).next[0]}")

final = graph.invoke(None, config)
print(f"  3. Resumed & done    -> Final status: {final['status']}")
