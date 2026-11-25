# Interrupts
> Venkata Bhattaram (c) 2026

## Contents
* Static breakpoints
* interrupt_before / interrupt_after
* Dynamic interrupt()

## Overview
Human-in-the-loop (HITL) workflows require a graph to pause safely, wait for external input, and resume right where it left off. LangGraph supports two complementary mechanisms (both require a checkpointer):

1. **Static breakpoints (`interrupt_before` / `interrupt_after`)**: configured at `builder.compile(checkpointer=..., interrupt_before=["publish"])`. The graph always pauses right before or after the named node, and resumes when you call `graph.invoke(None, config)`.
2. **Dynamic `interrupt(payload)`**: called inside a node function. It pauses execution, returns `payload` to the caller under `__interrupt__`, and receives the human's answer back as the return value of `interrupt()` when resumed with `graph.invoke(Command(resume=value), config)`.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 06-human-in-the-loop/code/interrupts.py
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


class ApprovalState(TypedDict):
    draft: str
    approved: bool
    status: str


def create_draft(state: ApprovalState) -> dict:
    return {"draft": "Release v2.0 deployment notes"}


def human_gate(state: ApprovalState) -> dict:
    decision = interrupt({"action": "approve_release", "draft": state["draft"]})
    return {"approved": bool(decision)}


def publish(state: ApprovalState) -> dict:
    return {"status": "published" if state["approved"] else "cancelled"}


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

paused_1 = graph.invoke({"draft": "", "approved": False, "status": ""}, config)
print("Paused at dynamic interrupt:", paused_1.get("__interrupt__"))

graph.invoke(Command(resume=True), config)
print("Paused before static breakpoint; next node:", graph.get_state(config).next)

final = graph.invoke(None, config)
print("Final state:", final)
```

## Conclusion
Use `interrupt_before`/`interrupt_after` for predictable debugging breakpoints and `interrupt()` + `Command(resume=...)` for runtime, data-driven human approval gates inside nodes.
