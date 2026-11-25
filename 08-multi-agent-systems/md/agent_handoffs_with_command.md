# Agent Handoffs with Command
> Venkata Bhattaram (c) 2026

## Contents
* Command(goto=...)
* Passing state between agents
* Direct agent-to-agent handoffs

## Overview
Traditionally, a LangGraph node returns a state update dict and a *separate* routing function inspects state to pick the next node. **`Command(update=..., goto=...)`** combines both in a single return value: an agent node can update state *and* immediately hand off control to another agent (or `END`).

Annotating the return type with `Command[Literal["billing_agent", "tech_agent"]]` lets LangGraph validate and visualize the possible handoff edges automatically without calling `add_conditional_edges`.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 08-multi-agent-systems/code/agent_handoffs_with_command.py
```

```python
import operator
from typing import Annotated, Literal, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class HandoffState(TypedDict):
    request: str
    trail: Annotated[list[str], operator.add]
    resolution: str


def triage_agent(state: HandoffState) -> Command[Literal["billing_agent", "tech_agent"]]:
    req = state["request"].lower()
    target = "billing_agent" if "refund" in req or "invoice" in req else "tech_agent"
    return Command(update={"trail": [f"triage_agent -> handing off to {target}"]}, goto=target)


def billing_agent(state: HandoffState) -> Command[Literal["__end__"]]:
    return Command(update={"trail": ["billing_agent -> issued refund #RF-209"], "resolution": "Refund processed."}, goto=END)


def tech_agent(state: HandoffState) -> Command[Literal["__end__"]]:
    return Command(update={"trail": ["tech_agent -> reset API token"], "resolution": "Technical issue resolved."}, goto=END)


builder = StateGraph(HandoffState)
builder.add_node("triage_agent", triage_agent)
builder.add_node("billing_agent", billing_agent)
builder.add_node("tech_agent", tech_agent)
builder.add_edge(START, "triage_agent")
graph = builder.compile()

print(graph.invoke({"request": "I need a refund on invoice #44", "trail": [], "resolution": ""}))
```

## Conclusion
`Command(update=..., goto=...)` makes agent handoffs concise and explicit—each agent updates shared state and transfers control in one atomic step.
