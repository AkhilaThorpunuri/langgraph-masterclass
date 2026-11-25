# Supervisor Pattern
> Venkata Bhattaram (c) 2026

## Contents
* Supervisor node routing
* Worker agents as nodes
* Aggregating worker output

## Overview
In the **Supervisor pattern**, a single coordinator node inspects the shared state, decides which specialist worker should run next (e.g., `sql_worker` or `chart_worker`), and routes back to `END` once all required work is complete. Every worker always returns control to the supervisor, which keeps routing logic centralized in one place instead of spread across every agent.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 08-multi-agent-systems/code/supervisor_pattern.py
```

```python
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
    return {"completed": ["sql_worker"], "outputs": ["SQL Worker: Q1=120k, Q2=145k, Q3=160k"]}


def chart_worker(state: SupervisorState) -> dict:
    return {"completed": ["chart_worker"], "outputs": ["Chart Worker: Generated revenue trend bar chart"]}


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("sql_worker", sql_worker)
builder.add_node("chart_worker", chart_worker)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda s: END if s["next_worker"] == "FINISH" else s["next_worker"],
    {"sql_worker": "sql_worker", "chart_worker": "chart_worker", END: END},
)
builder.add_edge("sql_worker", "supervisor")
builder.add_edge("chart_worker", "supervisor")
graph = builder.compile()

res = graph.invoke({"query": "Build quarterly revenue report", "completed": [], "outputs": [], "next_worker": ""})
print(res["outputs"])
```

## Conclusion
The Supervisor pattern is the cleanest default for multi-agent systems: workers stay focused on their single specialty, while the supervisor owns orchestration, sequencing, and completion criteria.
