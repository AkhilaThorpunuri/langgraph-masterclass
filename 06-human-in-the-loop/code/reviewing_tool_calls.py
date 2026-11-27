"""Reviewing Tool Calls -- approve, edit arguments, or reject a sensitive tool call.

Objective:
    Show how to gate a high-impact @tool call behind interrupt() so a human can
    approve it as-is, edit its arguments, or reject it completely.
"""
import socket
from typing import Any, TypedDict

import mlflow
from langchain_core.tools import tool
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


# STEP 1 -- TOOL & STATE.
@tool
def transfer_funds(to_account: str, amount: float) -> str:
    """Transfer funds to a target account."""
    return f"Transferred ${amount:.2f} to {to_account}"


class ToolReviewState(TypedDict):
    proposed_call: dict[str, Any]
    tool_result: str


# STEP 2 -- NODES.
def propose_tool_call(state: ToolReviewState) -> dict:
    # state coming IN:  {"proposed_call": {}, "tool_result": ""}
    # state going OUT: {"proposed_call": {"tool": "transfer_funds", "args": {...}}}
    return {
        "proposed_call": {
            "tool": "transfer_funds",
            "args": {"to_account": "ACCT-991", "amount": 5000.0},
        }
    }


def review_and_execute(state: ToolReviewState) -> dict:
    # Pauses with the proposed tool call; executes original, edited, or skips based on decision.
    call = state["proposed_call"]
    decision = interrupt({"message": "Approve, edit, or reject this tool call", "call": call})

    action = decision.get("action", "reject")
    if action == "approve":
        output = transfer_funds.invoke(call["args"])
    elif action == "edit":
        output = transfer_funds.invoke(decision["args"])
    else:
        output = "Tool call rejected by human reviewer."
    return {"tool_result": output}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(ToolReviewState)
builder.add_node("propose_tool_call", propose_tool_call)
builder.add_node("review_and_execute", review_and_execute)
builder.add_edge(START, "propose_tool_call")
builder.add_edge("propose_tool_call", "review_and_execute")
builder.add_edge("review_and_execute", END)

graph = builder.compile(checkpointer=MemorySaver())

# STEP 4 -- INVOKE TWO SCENARIOS (EDIT vs REJECT):
print("\n--- Case 1: Human Edits Tool Arguments ---")
cfg_edit = {"configurable": {"thread_id": "tool-review-edit"}}
p1 = graph.invoke({"proposed_call": {}, "tool_result": ""}, cfg_edit)
proposed = p1["__interrupt__"][0].value["call"]
print(f"  Proposed call: {proposed['tool']}({proposed['args']})")
r1 = graph.invoke(Command(resume={"action": "edit", "args": {"to_account": "ACCT-991", "amount": 500.0}}), cfg_edit)
print(f"  Result:        {r1['tool_result']}")

print("\n--- Case 2: Human Rejects Tool Call ---")
cfg_reject = {"configurable": {"thread_id": "tool-review-reject"}}
graph.invoke({"proposed_call": {}, "tool_result": ""}, cfg_reject)
r2 = graph.invoke(Command(resume={"action": "reject"}), cfg_reject)
print(f"  Result:        {r2['tool_result']}")
