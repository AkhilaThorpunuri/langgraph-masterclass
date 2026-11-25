"""Agent Handoffs with Command -- updating state AND routing in a single return value.

Demonstrates:
- Returning Command(update=..., goto=...) from an agent node
- Direct agent-to-agent handoffs without separate conditional edge routing functions
"""
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
    return Command(
        update={"trail": [f"triage_agent -> handing off to {target}"]},
        goto=target,
    )


def billing_agent(state: HandoffState) -> Command[Literal["__end__"]]:
    return Command(
        update={
            "trail": ["billing_agent -> issued refund #RF-209"],
            "resolution": "Refund processed.",
        },
        goto=END,
    )


def tech_agent(state: HandoffState) -> Command[Literal["__end__"]]:
    return Command(
        update={
            "trail": ["tech_agent -> reset API token"],
            "resolution": "Technical issue resolved.",
        },
        goto=END,
    )


builder = StateGraph(HandoffState)
builder.add_node("triage_agent", triage_agent)
builder.add_node("billing_agent", billing_agent)
builder.add_node("tech_agent", tech_agent)
builder.add_edge(START, "triage_agent")
graph = builder.compile()

for req in ["I need a refund on invoice #44", "My webhook keeps returning 502"]:
    res = graph.invoke({"request": req, "trail": [], "resolution": ""})
    print(f"Request: {req!r}")
    print("  Trail:", " | ".join(res["trail"]))
    print("  Resolution:", res["resolution"])
