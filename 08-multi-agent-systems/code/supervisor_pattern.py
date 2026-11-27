"""Supervisor Pattern -- a central supervisor routing work to specialist worker agents.

Objective:
    Show a hub-and-spoke multi-agent graph where a central `supervisor` node
    delegates to `sql_worker` and `chart_worker`, and every worker returns to
    `supervisor` until all tasks are complete.
"""
import operator
import socket
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph


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


# STEP 1 -- STATE: tracks which workers have completed and accumulates their outputs.
class SupervisorState(TypedDict):
    query: str
    completed: Annotated[list[str], operator.add]
    outputs: Annotated[list[str], operator.add]
    next_worker: str


# STEP 2 -- SUPERVISOR & WORKER NODES.
def supervisor(state: SupervisorState) -> dict:
    # Pass 1 IN: completed=[]                         -> OUT: next_worker="sql_worker"
    # Pass 2 IN: completed=["sql_worker"]             -> OUT: next_worker="chart_worker"
    # Pass 3 IN: completed=["sql_worker","chart_worker"] -> OUT: next_worker="FINISH"
    done = set(state.get("completed", []))
    if "sql_worker" not in done:
        return {"next_worker": "sql_worker"}
    if "chart_worker" not in done:
        return {"next_worker": "chart_worker"}
    return {"next_worker": "FINISH"}


def sql_worker(state: SupervisorState) -> dict:
    return {"completed": ["sql_worker"], "outputs": ["SQL Worker: Q1=120k, Q2=145k, Q3=160k"]}


def chart_worker(state: SupervisorState) -> dict:
    return {"completed": ["chart_worker"], "outputs": ["Chart Worker: Generated revenue trend bar chart"]}


# STEP 3 -- ROUTING FUNCTION & GRAPH WIRING.
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

# STEP 4 -- INVOKE: supervisor -> sql_worker -> supervisor -> chart_worker -> supervisor -> END.
print("\n--- Supervisor Pattern Demo ---")
res = graph.invoke({"query": "Build quarterly revenue report", "completed": [], "outputs": [], "next_worker": ""})
print(f"  Query: {res['query']}")
for idx, item in enumerate(res["outputs"], start=1):
    print(f"  Worker {idx} -> {item}")
