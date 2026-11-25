# Customer Support Agent
> Venkata Bhattaram (c) 2026

## Contents
* Requirements and design
* Knowledge base lookup
* Human escalation handoff

## Overview
A real-world customer support agent combines **conditional routing**, **knowledge-base lookup**, and **human-in-the-loop escalation**:
1. **`triage`** inspects the incoming message and routes routine questions (`self_serve`) to `kb_lookup`, while routing refunds or fraud disputes (`escalate`) to `human_handoff`.
2. **`kb_lookup`** resolves common FAQs immediately without human intervention.
3. **`human_handoff`** calls `interrupt()` to pause the thread and hand the ticket to a specialist, resuming via `Command(resume=...)` once the specialist writes a response.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 14-real-world-projects/code/customer_support_agent.py
```

```python
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
    return {"intent": "escalate" if ("refund" in msg or "fraud" in msg) else "self_serve"}


def kb_lookup(state: SupportState) -> dict:
    msg = state["customer_message"].lower()
    for key, answer in KB.items():
        if key in msg:
            return {"kb_answer": answer, "final_response": f"[KB Auto-Reply] {answer}"}
    return {"kb_answer": "", "final_response": "[KB Auto-Reply] Please check our help center."}


def human_handoff(state: SupportState) -> dict:
    agent_reply = interrupt({"reason": "High-priority refund/dispute", "customer_message": state["customer_message"]})
    return {"final_response": f"[Human Specialist] {agent_reply}"}


builder = StateGraph(SupportState)
builder.add_node("triage", triage)
builder.add_node("kb_lookup", kb_lookup)
builder.add_node("human_handoff", human_handoff)
builder.add_edge(START, "triage")
builder.add_conditional_edges(
    "triage",
    lambda s: "human_handoff" if s["intent"] == "escalate" else "kb_lookup",
    {"kb_lookup": "kb_lookup", "human_handoff": "human_handoff"},
)
builder.add_edge("kb_lookup", END)
builder.add_edge("human_handoff", END)
graph = builder.compile(checkpointer=MemorySaver())

cfg2 = {"configurable": {"thread_id": "ticket-102"}}
graph.invoke({"customer_message": "I need a refund for duplicate charge", "intent": "", "kb_answer": "", "final_response": ""}, cfg2)
print(graph.invoke(Command(resume="Approved full refund to original card."), cfg2)["final_response"])
```

## Conclusion
This architecture automates high-volume FAQ tickets while guaranteeing that sensitive refund and dispute cases always pause for a human specialist.
