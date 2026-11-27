"""Customer Support Agent -- KB lookup, policy triage, and human escalation handoff.

Objective:
    Combine conditional routing, knowledge-base lookup, and dynamic interrupt()
    so routine FAQ tickets auto-resolve while refund/fraud tickets escalate to a human.
"""
import socket
from typing import Literal, TypedDict

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

KB = {
    "password": "Go to Settings -> Security -> Reset Password.",
    "hours": "Support hours are Mon-Fri, 9am-6pm IST.",
}


# STEP 1 -- STATE.
class SupportState(TypedDict):
    customer_message: str
    intent: str
    kb_answer: str
    final_response: str


# STEP 2 -- NODES.
def triage(state: SupportState) -> dict:
    # state coming IN:  {"customer_message": "...", ...}
    # state going OUT:  {"intent": "escalate" | "self_serve"}
    msg = state["customer_message"].lower()
    return {"intent": "escalate" if ("refund" in msg or "fraud" in msg) else "self_serve"}


def kb_lookup(state: SupportState) -> dict:
    msg = state["customer_message"].lower()
    for key, answer in KB.items():
        if key in msg:
            return {"kb_answer": answer, "final_response": f"[KB Auto-Reply] {answer}"}
    return {"kb_answer": "", "final_response": "[KB Auto-Reply] Please check our help center."}


def human_handoff(state: SupportState) -> dict:
    # Pauses the thread until a human support specialist submits a reply via Command(resume=...).
    agent_reply = interrupt({
        "reason": "High-priority refund/dispute requires human specialist",
        "customer_message": state["customer_message"],
    })
    return {"final_response": f"[Human Specialist] {agent_reply}"}


# STEP 3 -- ROUTING FUNCTION & GRAPH WIRING.
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

# STEP 4 -- INVOKE BOTH TICKET PATHS:
print("\n--- Customer Support Agent Demo ---")
cfg1 = {"configurable": {"thread_id": "ticket-101"}}
r1 = graph.invoke({"customer_message": "How do I reset my password?", "intent": "", "kb_answer": "", "final_response": ""}, cfg1)
print(f"  Ticket 101 (FAQ)    -> {r1['final_response']}")

cfg2 = {"configurable": {"thread_id": "ticket-102"}}
paused = graph.invoke({"customer_message": "I need a refund for duplicate charge", "intent": "", "kb_answer": "", "final_response": ""}, cfg2)
reason = paused["__interrupt__"][0].value["reason"]
print(f"  Ticket 102 (Paused) -> {reason}")
r2 = graph.invoke(Command(resume="Approved full refund to original card."), cfg2)
print(f"  Ticket 102 (Done)   -> {r2['final_response']}")
