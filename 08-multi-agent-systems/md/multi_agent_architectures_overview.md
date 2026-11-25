# Multi-Agent Architectures Overview
> Venkata Bhattaram (c) 2026

## Contents
* Single agent vs multi-agent
* Network, supervisor, and hierarchical patterns

## Overview
A single agent with dozens of tools and a giant system prompt quickly loses focus. **Multi-agent architectures** split responsibilities across focused specialist nodes—each with its own prompt, tools, and clear job:
* **Network (peer-to-peer)**: each specialist decides which peer should run next (e.g., `researcher -> writer -> reviewer`).
* **Supervisor**: a central coordinator delegates work to specialist workers and aggregates their results.
* **Hierarchical teams**: multiple specialist teams (each its own compiled subgraph) report to a top-level supervisor.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 08-multi-agent-systems/code/multi_agent_architectures_overview.py
```

```python
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
    return {"notes": [f"[Researcher] Gathered facts for: {state['task']}"], "next_agent": "writer"}


def writer(state: NetworkState) -> dict:
    return {"notes": ["[Writer] Drafted summary from research notes."], "next_agent": "reviewer"}


def reviewer(state: NetworkState) -> dict:
    return {"notes": ["[Reviewer] Approved final summary."], "next_agent": END}


builder = StateGraph(NetworkState)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
builder.add_edge(START, "researcher")
builder.add_conditional_edges("researcher", lambda s: s["next_agent"], {"writer": "writer", END: END})
builder.add_conditional_edges("writer", lambda s: s["next_agent"], {"reviewer": "reviewer", END: END})
builder.add_conditional_edges("reviewer", lambda s: s["next_agent"], {END: END})
graph = builder.compile()

print(graph.invoke({"task": "Explain LangGraph multi-agent patterns", "notes": [], "next_agent": ""})["notes"])
```

## Conclusion
Splitting complex workflows into specialist agent nodes keeps prompts small, tool sets focused, and execution traces easy to read.
