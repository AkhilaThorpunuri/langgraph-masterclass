"""Interrupts -- static breakpoints (interrupt_before) and dynamic interrupt().

Demonstrates:
- Static breakpoint with interrupt_before=["publish"]
- Dynamic interrupt() inside a node that pauses execution and resumes via Command(resume=...)
"""
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class ApprovalState(TypedDict):
    draft: str
    approved: bool
    status: str


def create_draft(state: ApprovalState) -> dict:
    return {"draft": "Release v2.0 deployment notes"}


def human_gate(state: ApprovalState) -> dict:
    # Dynamic interrupt: pauses here and surfaces the payload to the caller
    decision = interrupt({"action": "approve_release", "draft": state["draft"]})
    return {"approved": bool(decision)}


def publish(state: ApprovalState) -> dict:
    status = "published" if state["approved"] else "cancelled"
    return {"status": status}


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

# 1. Runs create_draft -> pauses inside human_gate via dynamic interrupt()
paused_1 = graph.invoke({"draft": "", "approved": False, "status": ""}, config)
print("Paused at dynamic interrupt:", paused_1.get("__interrupt__"))

# 2. Resume dynamic interrupt with approval=True -> hits static interrupt_before=['publish']
graph.invoke(Command(resume=True), config)
print("Paused before static breakpoint; next node:", graph.get_state(config).next)

# 3. Continue past static breakpoint to finish
final = graph.invoke(None, config)
print("Final state:", final)
