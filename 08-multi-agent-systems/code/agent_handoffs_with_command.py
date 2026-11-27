"""Agent Handoffs with Command -- updating state AND routing in a single return value.

Objective:
    Show how Command(update=..., goto=...) lets an agent node update state and
    hand off control directly to another agent without add_conditional_edges().
"""
import operator
import socket
from typing import Annotated, Literal, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


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
class HandoffState(TypedDict):
    request: str
    trail: Annotated[list[str], operator.add]
    resolution: str


# STEP 2 -- AGENT NODES RETURNING Command(update=..., goto=...):
# The return type annotation Command[Literal[...]] declares the allowed handoff targets.
def triage_agent(state: HandoffState) -> Command[Literal["billing_agent", "tech_agent"]]:
    # state coming IN:  {"request": "I need a refund...", "trail": [], "resolution": ""}
    # Command going OUT: appends handoff trail AND jumps straight to billing_agent or tech_agent
    req = state["request"].lower()
    target = "billing_agent" if "refund" in req or "invoice" in req else "tech_agent"
    return Command(
        update={"trail": [f"triage_agent -> handing off to {target}"]},
        goto=target,
    )


def billing_agent(state: HandoffState) -> Command[Literal["__end__"]]:
    return Command(
        update={"trail": ["billing_agent -> issued refund #RF-209"], "resolution": "Refund processed."},
        goto=END,
    )


def tech_agent(state: HandoffState) -> Command[Literal["__end__"]]:
    return Command(
        update={"trail": ["tech_agent -> reset API token"], "resolution": "Technical issue resolved."},
        goto=END,
    )


# STEP 3 -- GRAPH & COMPILE: notice NO add_conditional_edges() needed!
builder = StateGraph(HandoffState)
builder.add_node("triage_agent", triage_agent)
builder.add_node("billing_agent", billing_agent)
builder.add_node("tech_agent", tech_agent)
builder.add_edge(START, "triage_agent")
graph = builder.compile()

# STEP 4 -- INVOKE TWO DIFFERENT REQUESTS:
print("\n--- Agent Handoffs with Command ---")
for req in ["I need a refund on invoice #44", "My webhook keeps returning 502"]:
    res = graph.invoke({"request": req, "trail": [], "resolution": ""})
    print(f"\n  Request:    {req}")
    print(f"  Handoff:    {' -> '.join(res['trail'])}")
    print(f"  Resolution: {res['resolution']}")
