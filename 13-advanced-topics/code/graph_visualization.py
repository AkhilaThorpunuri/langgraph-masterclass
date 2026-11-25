"""Graph Visualization -- Mermaid diagram export and ASCII topology inspection.

Demonstrates:
- Inspecting compiled graph topology with graph.get_graph().draw_mermaid()
- Printing a Mermaid flowchart you can paste directly into Markdown or GitHub
"""
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class VizState(TypedDict):
    score: int
    status: str


def evaluate(state: VizState) -> dict:
    return {}


def approve(state: VizState) -> dict:
    return {"status": "approved"}


def reject(state: VizState) -> dict:
    return {"status": "rejected"}


def route(state: VizState) -> str:
    return "approve" if state["score"] >= 70 else "reject"


builder = StateGraph(VizState)
builder.add_node("evaluate", evaluate)
builder.add_node("approve", approve)
builder.add_node("reject", reject)
builder.add_edge(START, "evaluate")
builder.add_conditional_edges("evaluate", route, {"approve": "approve", "reject": "reject"})
builder.add_edge("approve", END)
builder.add_edge("reject", END)
graph = builder.compile()

print("=== Mermaid Diagram ===")
print(graph.get_graph().draw_mermaid())

print("=== Sample Run ===")
print(graph.invoke({"score": 85, "status": ""}))
