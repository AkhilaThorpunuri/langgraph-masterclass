"""Supervisor Pattern -- a central supervisor routing work to specialist worker agents.

Demonstrates:
- Central supervisor node inspecting progress and picking the next worker
- Worker agents returning results to the supervisor until FINISH
"""
import operator
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class SupervisorState(TypedDict):
    query: str
    completed: Annotated[list[str], operator.add]
    outputs: Annotated[list[str], operator.add]
    next_worker: str


def supervisor(state: SupervisorState) -> dict:
    done = set(state.get("completed", []))
    if "sql_worker" not in done:
        return {"next_worker": "sql_worker"}
    if "chart_worker" not in done:
        return {"next_worker": "chart_worker"}
    return {"next_worker": "FINISH"}


def sql_worker(state: SupervisorState) -> dict:
    return {
        "completed": ["sql_worker"],
        "outputs": ["SQL Worker: Q1=120k, Q2=145k, Q3=160k"],
    }


def chart_worker(state: SupervisorState) -> dict:
    return {
        "completed": ["chart_worker"],
        "outputs": ["Chart Worker: Generated revenue trend bar chart"],
    }


def route_supervisor(state: SupervisorState) -> str:
    return END if state["next_worker"] == "FINISH" else state["next_worker"]


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("sql_worker", sql_worker)
builder.add_node("chart_worker", chart_worker)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {"sql_worker": "sql_worker", "chart_worker": "chart_worker", END: END},
)
builder.add_edge("sql_worker", "supervisor")
builder.add_edge("chart_worker", "supervisor")
graph = builder.compile()

res = graph.invoke({"query": "Build quarterly revenue report", "completed": [], "outputs": [], "next_worker": ""})
print("Aggregated worker outputs:")
for item in res["outputs"]:
    print(" -", item)
