"""Multi-Agent Architectures Overview -- network-style peer routing between specialists.

Demonstrates:
- Multiple specialist agent nodes sharing one state
- Decentralized handoff where each specialist decides whether another specialist is needed
"""
import operator
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class NetworkState(TypedDict):
    task: str
    notes: Annotated[list[str], operator.add]
    next_agent: str


def researcher(state: NetworkState) -> dict:
    return {
        "notes": [f"[Researcher] Gathered facts for: {state['task']}"],
        "next_agent": "writer",
    }


def writer(state: NetworkState) -> dict:
    return {
        "notes": ["[Writer] Drafted summary from research notes."],
        "next_agent": "reviewer",
    }


def reviewer(state: NetworkState) -> dict:
    return {
        "notes": ["[Reviewer] Approved final summary."],
        "next_agent": END,
    }


def route_peer(state: NetworkState) -> str:
    return state["next_agent"]


builder = StateGraph(NetworkState)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "researcher")
builder.add_conditional_edges("researcher", route_peer, {"writer": "writer", END: END})
builder.add_conditional_edges("writer", route_peer, {"reviewer": "reviewer", END: END})
builder.add_conditional_edges("reviewer", route_peer, {END: END})
graph = builder.compile()

result = graph.invoke({"task": "Explain LangGraph multi-agent patterns", "notes": [], "next_agent": ""})
for line in result["notes"]:
    print(line)
