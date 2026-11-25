# Approving and Editing State
> Venkata Bhattaram (c) 2026

## Contents
* Pausing for review
* Updating state before resuming
* Resuming with Command

## Overview
Often a human reviewer does not just want to click "Approve" or "Reject"—they want to **edit** what the agent drafted before it takes an external action (like sending an email or publishing a message).

With `interrupt()`, a node surfaces the current draft to the caller. When the human submits an edited version via `Command(resume={"edited_body": ...})`, the node writes the edited value into state and downstream nodes operate on the human-approved version seamlessly.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 06-human-in-the-loop/code/approving_and_editing_state.py
```

```python
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class EmailState(TypedDict):
    recipient: str
    body: str
    sent: bool


def draft_email(state: EmailState) -> dict:
    return {"body": f"Hi {state['recipient']}, your invoice is overdue!"}


def review_email(state: EmailState) -> dict:
    review = interrupt({
        "prompt": "Review or edit email before sending",
        "current_body": state["body"],
    })
    return {"body": review.get("edited_body", state["body"])}


def send_email(state: EmailState) -> dict:
    print(f"Sending email to {state['recipient']}: {state['body']}")
    return {"sent": True}


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

paused = graph.invoke({"recipient": "alex@example.com", "body": "", "sent": False}, config)
print("Waiting for review:", paused.get("__interrupt__"))

final = graph.invoke(
    Command(resume={"edited_body": "Hi alex@example.com, friendly reminder that your invoice is due soon."}),
    config,
)
print("Final state:", final)
```

## Conclusion
Combining `interrupt()` with structured resume payloads turns any LangGraph node into an interactive review-and-edit checkpoint, ensuring humans stay in full control of outgoing actions.
