# Custom Reducers and Channels
> Venkata Bhattaram (c) 2026

## Contents
* Writing a custom reducer function
* Deduplicating list merges
* Merging partial dictionaries from parallel branches

## Overview
By default, returning a key from a node overwrites the previous value in state, and `Annotated[list, operator.add]` simply concatenates lists. When parallel branches write to the same field and you need richer merge semantics—such as **deduplicating tags** or **shallow-merging metrics dictionaries**—pass any custom two-argument function `(left, right) -> merged` inside `Annotated[T, my_reducer]`.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 13-advanced-topics/code/custom_reducers_and_channels.py
```

```python
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


def merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen = set(left)
    return left + [item for item in right if item not in seen]


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class CustomReducerState(TypedDict):
    tags: Annotated[list[str], merge_unique]
    metrics: Annotated[dict[str, float], merge_dicts]


def branch_a(state: CustomReducerState) -> dict:
    return {"tags": ["python", "langgraph"], "metrics": {"latency_a_ms": 12.4}}


def branch_b(state: CustomReducerState) -> dict:
    return {"tags": ["langgraph", "mlflow"], "metrics": {"latency_b_ms": 18.7}}


builder = StateGraph(CustomReducerState)
builder.add_node("branch_a", branch_a)
builder.add_node("branch_b", branch_b)
builder.add_edge(START, "branch_a")
builder.add_edge(START, "branch_b")
builder.add_edge("branch_a", END)
builder.add_edge("branch_b", END)
graph = builder.compile()

print(graph.invoke({"tags": [], "metrics": {}}))
```

## Conclusion
Custom reducers keep parallel fan-out/fan-in graphs conflict-free and deterministic without cluttering your node functions with manual merge logic.
