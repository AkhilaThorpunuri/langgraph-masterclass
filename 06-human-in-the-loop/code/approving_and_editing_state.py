"""Approving and Editing State -- pausing for human review, editing state, and resuming.

Objective:
    Show how a human can inspect an agent's draft, rewrite it, and inject the
    edited text back into state via Command(resume=...) before an external action.
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
class EmailState(TypedDict):
    recipient: str
    body: str
    sent: bool


# STEP 2 -- NODES.
def draft_email(state: EmailState) -> dict:
    # state coming IN:  {"recipient": "alex@example.com", "body": "", "sent": False}
    # state going OUT: {"body": "Hi alex@example.com, your invoice is overdue!"}
    return {"body": f"Hi {state['recipient']}, your invoice is overdue!"}


def review_email(state: EmailState) -> dict:
    # Pauses and surfaces `current_body`; overwrites `body` with human's edit on resume.
    review = interrupt({
        "prompt": "Review or edit email before sending",
        "current_body": state["body"],
    })
    return {"body": review.get("edited_body", state["body"])}


def send_email(state: EmailState) -> dict:
    # state coming IN:  {"body": "<human-approved text>", ...}
    # state going OUT: {"sent": True}
    return {"sent": True}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(EmailState)
builder.add_node("draft_email", draft_email)
builder.add_node("review_email", review_email)
builder.add_node("send_email", send_email)
builder.add_edge(START, "draft_email")
builder.add_edge("draft_email", "review_email")
builder.add_edge("review_email", "send_email")
builder.add_edge("send_email", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "email-review-1"}}

# STEP 4 -- PAUSE FOR REVIEW, THEN RESUME WITH HUMAN EDIT:
print("\n--- Approving & Editing State Demo ---")
paused = graph.invoke({"recipient": "alex@example.com", "body": "", "sent": False}, config)
review_info = paused["__interrupt__"][0].value
print(f"  Original draft: {review_info['current_body']}")

final = graph.invoke(
    Command(resume={"edited_body": "Hi alex@example.com, friendly reminder that your invoice is due soon."}),
    config,
)
print(f"  Edited & sent:  {final['body']}")
print(f"  Sent status:    {final['sent']}")
