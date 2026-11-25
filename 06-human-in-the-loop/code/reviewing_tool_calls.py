"""Reviewing Tool Calls -- approve, edit arguments, or reject a sensitive tool call.

Demonstrates:
- Human-in-the-loop review before executing a high-impact tool call
- Handling three decisions: approve, edit (args), or reject
"""
from typing import Any, TypedDict

import mlflow
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


@tool
def transfer_funds(to_account: str, amount: float) -> str:
    """Transfer funds to a target account."""
    return f"Transferred ${amount:.2f} to {to_account}"


class ToolReviewState(TypedDict):
    proposed_call: dict[str, Any]
    tool_result: str


def propose_tool_call(state: ToolReviewState) -> dict:
    return {
        "proposed_call": {
            "tool": "transfer_funds",
            "args": {"to_account": "ACCT-991", "amount": 5000.0},
        }
    }


def review_and_execute(state: ToolReviewState) -> dict:
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


builder = StateGraph(ToolReviewState)
builder.add_node("propose_tool_call", propose_tool_call)
builder.add_node("review_and_execute", review_and_execute)
builder.add_edge(START, "propose_tool_call")
builder.add_edge("propose_tool_call", "review_and_execute")
builder.add_edge("review_and_execute", END)

graph = builder.compile(checkpointer=MemorySaver())

# Case 1: Human edits the transfer amount before executing
cfg_edit = {"configurable": {"thread_id": "tool-review-edit"}}
print("Paused (edit case):", graph.invoke({"proposed_call": {}, "tool_result": ""}, cfg_edit).get("__interrupt__"))
print("After edit:", graph.invoke(Command(resume={"action": "edit", "args": {"to_account": "ACCT-991", "amount": 500.0}}), cfg_edit))

# Case 2: Human rejects the tool call
cfg_reject = {"configurable": {"thread_id": "tool-review-reject"}}
graph.invoke({"proposed_call": {}, "tool_result": ""}, cfg_reject)
print("After reject:", graph.invoke(Command(resume={"action": "reject"}), cfg_reject))
