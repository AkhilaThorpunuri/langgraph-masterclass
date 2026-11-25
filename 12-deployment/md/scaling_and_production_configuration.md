# Scaling and Production Configuration
> Venkata Bhattaram (c) 2026

## Contents
* Choosing a production checkpointer
* Concurrency and batch execution
* Recursion limits and latency monitoring

## Overview
Moving a LangGraph prototype into production requires three core safeguards:
1. **Shared persistence**: replace `MemorySaver` with `PostgresSaver` so multiple horizontally scaled workers share thread state.
2. **Bounded concurrency**: use `graph.batch(inputs, config=configs, max_concurrency=N)` to throttle parallel load against rate-limited LLM APIs.
3. **Safety caps & telemetry**: set `recursion_limit` per request to prevent runaway loops and track end-to-end latency in MLflow.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 12-deployment/code/scaling_and_production_configuration.py
```

```python
import time
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class WorkState(TypedDict):
    item_id: int
    result: str


def process_item(state: WorkState) -> dict:
    return {"result": f"processed-item-{state['item_id']}"}


builder = StateGraph(WorkState)
builder.add_node("process_item", process_item)
builder.add_edge(START, "process_item")
builder.add_edge("process_item", END)
graph = builder.compile(checkpointer=MemorySaver())

inputs = [{"item_id": i, "result": ""} for i in range(1, 6)]
configs = [{"configurable": {"thread_id": f"prod-req-{i}"}, "recursion_limit": 10} for i in range(1, 6)]

t0 = time.perf_counter()
results = graph.batch(inputs, config=configs, max_concurrency=2)
print(f"Processed {len(results)} requests in {(time.perf_counter() - t0) * 1000:.1f} ms")
```

## Conclusion
Always bound concurrency (`max_concurrency`), cap loop iterations (`recursion_limit`), and externalize checkpoints to PostgreSQL when scaling beyond a single worker process.
