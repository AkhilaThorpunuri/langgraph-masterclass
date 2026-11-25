"""Customer Support Agent -- KB lookup, policy tools, and human escalation handoff.

Demonstrates:
- Triage node classifying a support ticket (auto-resolvable vs needs human escalation)
- Knowledge-base lookup for common questions
- Dynamic interrupt() handoff to a human support specialist for refunds/disputes
"""
from typing import Literal, TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()

KB = {
    "password": "Go to Settings -> Security -> Reset Password.",
    "hours": "Support hours are Mon-Fri, 9am-6pm IST.",
}


class SupportState(TypedDict):
    customer_message: str
    intent: str
    kb_answer: str
    final_response: str


def triage(state: SupportState) -> dict:
    msg = state["customer_message"].lower()
    if "refund" in msg or "fraud" in msg:
        return {"intent": "escalate"}
    return {"intent": "self_serve"}


def kb_lookup(state: SupportState) -> dict:
    msg = state["customer_message"].lower()
    for key, answer in KB.items():
        if key in msg:
            return {"kb_answer": answer, "final_response": f"[KB Auto-Reply] {answer}"}
    return {"kb_answer": "", "final_response": "[KB Auto-Reply] Please check our help center."}


def human_handoff(state: SupportState) -> dict:
    agent_reply = interrupt({
        "reason": "High-priority refund/dispute requires human specialist",
        "customer_message": state["customer_message"],
    })
    return {"final_response": f"[Human Specialist] {agent_reply}"}


def route_triage(state: SupportState) -> Literal["kb_lookup", "human_handoff"]:
    return "human_handoff" if state["intent"] == "escalate" else "kb_lookup"


builder = StateGraph(SupportState)
builder.add_node("triage", triage)
builder.add_node("kb_lookup", kb_lookup)
builder.add_node("human_handoff", human_handoff)

builder.add_edge(START, "triage")
builder.add_conditional_edges(
    "triage",
    route_triage,
    {"kb_lookup": "kb_lookup", "human_handoff": "human_handoff"},
)
builder.add_edge("kb_lookup", END)
builder.add_edge("human_handoff", END)
graph = builder.compile(checkpointer=MemorySaver())

# 1. Self-serve KB ticket
cfg1 = {"configurable": {"thread_id": "ticket-101"}}
r1 = graph.invoke({"customer_message": "How do I reset my password?", "intent": "", "kb_answer": "", "final_response": ""}, cfg1)
print("Ticket 101:", r1["final_response"])

# 2. Refund ticket -> pauses for human specialist -> resumes
cfg2 = {"configurable": {"thread_id": "ticket-102"}}
paused = graph.invoke({"customer_message": "I need a refund for duplicate charge", "intent": "", "kb_answer": "", "final_response": ""}, cfg2)
print("Ticket 102 paused:", paused.get("__interrupt__"))
r2 = graph.invoke(Command(resume="Approved full refund to original card."), cfg2)
print("Ticket 102 resolved:", r2["final_response"])
