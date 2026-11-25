# LangGraph Studio
> Venkata Bhattaram (c) 2026

## Contents
* Visualizing graph runs
* Stepping through execution
* Editing state mid-run

## Overview
**LangGraph Studio** connects to a compiled `graph` (served locally via `langgraph dev`) and provides a visual debugger for your agents. Under the hood, Studio relies on the exact same checkpoint primitives you use in Python:
* **Breakpoints** (`interrupt_before` / `interrupt_after`) to pause before any node.
* **State inspection** (`graph.get_state(config)`) to view every key at the pause point.
* **Mid-run editing** (`graph.update_state(config, patch)`) to tweak state and continue executing.

## Code Example
Run the standalone script (simulates a Studio pause-and-edit session with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 11-testing-debugging-and-observability/code/langgraph_studio.py
```

```python
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class StudioState(TypedDict):
    query: str
    plan: str
    result: str


def plan_node(state: StudioState) -> dict:
    return {"plan": f"Step 1: research {state['query']}"}


def execute_node(state: StudioState) -> dict:
    return {"result": f"Executed -> {state['plan']}"}


builder = StateGraph(StudioState)
builder.add_node("plan_node", plan_node)
builder.add_node("execute_node", execute_node)
builder.add_edge(START, "plan_node")
builder.add_edge("plan_node", "execute_node")
builder.add_edge("execute_node", END)

graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["execute_node"])

cfg = {"configurable": {"thread_id": "studio-step-1"}}
graph.invoke({"query": "observability", "plan": "", "result": ""}, cfg)
print("Paused before:", graph.get_state(cfg).next)
graph.update_state(cfg, {"plan": "Step 1: research observability + MLflow tracing"})
print("Resumed final state:", graph.invoke(None, cfg))
```

## Conclusion
Understanding how `interrupt_before` and `update_state` work programmatically makes LangGraph Studio intuitive—everything you click in the Studio UI maps directly to these checkpoint APIs.
