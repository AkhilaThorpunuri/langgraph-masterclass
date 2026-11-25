# Introduction to LangGraph
> Venkata Bhattaram (c) 2026

## Contents
* Why LangGraph exists
* Stateful graphs vs linear chains
* What you will build in this course

## Overview
**LangGraph** is a low-level orchestration framework for building stateful, multi-step AI agents and workflows as directed graphs. Unlike simple linear chains, real-world agents need **cycles** (tool-calling loops and retries), **conditional branching**, **persistent state** (checkpoints and memory), and **human-in-the-loop** pauses.

Every LangGraph application is built from three core primitives:
1. **State**: a typed dictionary (or Pydantic model) shared across every step.
2. **Nodes**: plain Python functions that read state and return partial updates.
3. **Edges**: fixed or conditional connections that control which node runs next—including loops and parallel branches.

## Code Example
Run the starter demo in `01-getting-started/code/basics_demo.py`:
```powershell
python 01-getting-started/code/basics_demo.py
```

```python
from typing import TypedDict
from langgraph.graph import END, START, StateGraph


class HelloState(TypedDict):
    name: str
    greeting: str


def greet(state: HelloState) -> dict:
    return {"greeting": f"Welcome to LangGraph, {state['name']}!"}


builder = StateGraph(HelloState)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)
graph = builder.compile()

print(graph.invoke({"name": "Developer", "greeting": ""}))
```

## Conclusion
LangGraph gives you explicit, debuggable control over agent execution—from a one-node greeting graph here all the way to multi-agent systems, RAG pipelines, and production deployments in chapters `02` through `14`.
