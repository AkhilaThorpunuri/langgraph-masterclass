# Compiling and Invoking a Graph
> Venkata Bhattaram (c) 2026

## Contents
* `builder.compile()`
* `invoke` vs `stream` vs `batch`
* `config` and `recursion_limit`

## Overview
A `StateGraph` builder only describes nodes and edges; calling **`builder.compile()`** validates the topology and produces an executable `CompiledStateGraph`. Once compiled, you can run it in three ways:
* **`graph.invoke(input, config)`**: runs to completion and returns the final state dict.
* **`graph.stream(input, stream_mode="updates")`**: yields each node's output as it finishes.
* **`graph.batch([input1, input2], config=...)`**: processes multiple inputs concurrently.

Use `config={"recursion_limit": N}` (default `25`) to cap the maximum number of supersteps and protect cyclic graphs against infinite loops.

## Code Example
```python
from typing import TypedDict
from langgraph.graph import END, START, StateGraph


class DoubleState(TypedDict):
    n: int
    doubled: int


def double(state: DoubleState) -> dict:
    return {"doubled": state["n"] * 2}


builder = StateGraph(DoubleState)
builder.add_node("double", double)
builder.add_edge(START, "double")
builder.add_edge("double", END)
graph = builder.compile()

# 1. Single synchronous run
print("invoke:", graph.invoke({"n": 5, "doubled": 0}, config={"recursion_limit": 10}))

# 2. Streaming node updates
for update in graph.stream({"n": 5, "doubled": 0}, stream_mode="updates"):
    print("stream:", update)

# 3. Batched execution
print("batch:", graph.batch([{"n": 2, "doubled": 0}, {"n": 8, "doubled": 0}]))
```

## Conclusion
`compile()` locks in your graph structure and unlocks `invoke`, `stream`, `batch`, checkpointers, and `recursion_limit` configuration—the runtime foundation used across every remaining chapter.
