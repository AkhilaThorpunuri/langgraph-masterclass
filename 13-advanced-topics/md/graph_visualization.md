# Graph Visualization
> Venkata Bhattaram (c) 2026

## Contents
* Mermaid diagrams
* PNG export
* Debugging graph topology visually

## Overview
As graphs grow to include conditional branches, loops, and subgraphs, visualizing the compiled topology catches wiring mistakes immediately. Every compiled LangGraph app exposes `graph.get_graph()`, which can render:
* **Mermaid syntax** (`graph.get_graph().draw_mermaid()`)—paste directly into Markdown, GitHub, or documentation.
* **PNG image** (`graph.get_graph().draw_mermaid_png()`)—embed in Jupyter notebooks or save to disk.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 13-advanced-topics/code/graph_visualization.py
```

```python
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class VizState(TypedDict):
    score: int
    status: str


builder = StateGraph(VizState)
builder.add_node("evaluate", lambda s: {})
builder.add_node("approve", lambda s: {"status": "approved"})
builder.add_node("reject", lambda s: {"status": "rejected"})
builder.add_edge(START, "evaluate")
builder.add_conditional_edges("evaluate", lambda s: "approve" if s["score"] >= 70 else "reject", {"approve": "approve", "reject": "reject"})
builder.add_edge("approve", END)
builder.add_edge("reject", END)
graph = builder.compile()

print(graph.get_graph().draw_mermaid())
print(graph.invoke({"score": 85, "status": ""}))
```

## Conclusion
Use `draw_mermaid()` during development and in pull requests so teammates can verify every conditional edge and termination path at a glance.
