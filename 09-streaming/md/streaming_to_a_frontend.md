# Streaming to a Frontend
> Venkata Bhattaram (c) 2026

## Contents
* Server-sent events (SSE)
* FastAPI StreamingResponse
* Integrating with a browser UI

## Overview
To display live LangGraph progress in a web frontend, wrap `graph.stream(..., stream_mode="updates")` inside a **FastAPI `StreamingResponse`** emitting **Server-Sent Events (`text/event-stream`)**. Each node update is JSON-encoded as `data: {...}\n\n`, which any browser can consume natively with `EventSource` or `fetch()`.

## Code Example
Run the standalone script (simulates SSE output directly, or serve live with `uvicorn streaming_to_a_frontend:app --reload`):
```powershell
python 09-streaming/code/streaming_to_a_frontend.py
```

```python
import json
from typing import TypedDict

import mlflow
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class JobState(TypedDict):
    topic: str
    outline: str
    article: str


def make_outline(state: JobState) -> dict:
    return {"outline": f"1. Intro to {state['topic']}\n2. Key Patterns"}


def write_article(state: JobState) -> dict:
    return {"article": f"Full article built from:\n{state['outline']}"}


builder = StateGraph(JobState)
builder.add_node("make_outline", make_outline)
builder.add_node("write_article", write_article)
builder.add_edge(START, "make_outline")
builder.add_edge("make_outline", "write_article")
builder.add_edge("write_article", END)
graph = builder.compile()

app = FastAPI(title="LangGraph SSE Streaming Demo")


@app.get("/stream")
def stream_graph(topic: str = "LangGraph"):
    def event_generator():
        for update in graph.stream({"topic": topic, "outline": "", "article": ""}, stream_mode="updates"):
            yield f"data: {json.dumps(update)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

## Conclusion
Server-Sent Events over FastAPI are the simplest, most reliable way to push live LangGraph node updates to a web UI without managing WebSocket connection state.
