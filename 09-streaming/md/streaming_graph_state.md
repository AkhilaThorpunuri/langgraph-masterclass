# Streaming Graph State
> Venkata Bhattaram (c) 2026

## Contents
* stream() vs invoke()
* Stream modes: values, updates, debug

## Overview
`graph.invoke()` waits until the entire graph finishes before returning the final state. For long-running or multi-step graphs, **`graph.stream()`** yields intermediate results after every node completes, supporting three core modes:
* **`updates`**: yields only `{node_name: partial_update}` for each node as it finishes (most compact and UI-friendly).
* **`values`**: yields the full merged state dictionary after every superstep.
* **`debug`**: yields detailed internal metadata including task IDs, step numbers, and checkpoints.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 09-streaming/code/streaming_graph_state.py
```

```python
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class StepState(TypedDict):
    text: str
    steps: list[str]


def clean(state: StepState) -> dict:
    return {"text": state["text"].strip().lower(), "steps": state["steps"] + ["clean"]}


def summarize(state: StepState) -> dict:
    return {"text": f"summary({state['text']})", "steps": state["steps"] + ["summarize"]}


builder = StateGraph(StepState)
builder.add_node("clean", clean)
builder.add_node("summarize", summarize)
builder.add_edge(START, "clean")
builder.add_edge("clean", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

initial = {"text": "  LangGraph Streaming  ", "steps": []}
for chunk in graph.stream(initial, stream_mode="updates"):
    print("update:", chunk)
for chunk in graph.stream(initial, stream_mode="values"):
    print("value: ", chunk)
```

## Conclusion
Choose `stream_mode="updates"` to show step-by-step progress indicators, `"values"` when your consumer needs the complete state snapshot, and `"debug"` when diagnosing execution internals.
