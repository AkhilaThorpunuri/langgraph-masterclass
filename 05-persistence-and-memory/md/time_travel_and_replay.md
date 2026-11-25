# Time Travel and Replay
> Venkata Bhattaram (c) 2026

## Contents
* Viewing checkpoint history
* Replaying from a past state
* Forking a run

## Overview
Because a checkpointer saves a snapshot of state after every superstep, LangGraph lets you inspect the full timeline of any thread and **time travel** back to an earlier point in execution.

* **`graph.get_state_history(config)`** returns every checkpoint for a `thread_id` in reverse chronological order, including the state values and the `next` node(s) scheduled to run.
* **Replaying** invokes the graph with `None` as input and a historical checkpoint's `config` (containing its `checkpoint_id`).
* **Forking** calls `graph.update_state(past_config, new_values)` to create a brand-new branch from that historical checkpoint with modified state—without overwriting the original history—and then resumes from the forked config.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 05-persistence-and-memory/code/time_travel_and_replay.py
```

```python
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class PipelineState(TypedDict):
    topic: str
    draft: str
    polished: str


def write_draft(state: PipelineState) -> dict:
    return {"draft": f"Initial draft about {state['topic']}"}


def polish_draft(state: PipelineState) -> dict:
    return {"polished": f"[POLISHED] {state['draft']}"}


builder = StateGraph(PipelineState)
builder.add_node("write_draft", write_draft)
builder.add_node("polish_draft", polish_draft)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "polish_draft")
builder.add_edge("polish_draft", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "time-travel-demo"}}

original = graph.invoke({"topic": "LangGraph", "draft": "", "polished": ""}, config)
print("Original run:", original)

history = list(graph.get_state_history(config))
before_polish = next(s for s in history if s.next == ("polish_draft",))

fork_config = graph.update_state(
    before_polish.config,
    {"draft": "Forked draft about LangGraph Time Travel"},
)
replayed = graph.invoke(None, fork_config)
print("Forked & replayed run:", replayed)
```

## Conclusion
Time travel turns debugging and "what-if" exploration into first-class operations: you can inspect any intermediate checkpoint, edit state at that exact step, and replay only the downstream nodes from the forked checkpoint.
