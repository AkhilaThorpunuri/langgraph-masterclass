# Dockerizing a Graph
> Venkata Bhattaram (c) 2026

## Contents
* FastAPI wrapper for a LangGraph app
* Environment variables and secrets
* Dockerfile template

## Overview
When self-hosting a LangGraph workflow in Docker, Kubernetes, or Cloud Run, wrap your compiled graph inside a lightweight **FastAPI** app exposing `/health` (for liveness/readiness probes) and `/invoke` (validated with Pydantic). Pass API keys and environment flags (`APP_ENV`, database URLs) via runtime environment variables—never bake secrets into the container image.

## Code Example
Run the standalone script (prints the Dockerfile template and runs a local smoke test with MLflow tracing):
```powershell
python 12-deployment/code/dockerizing_a_graph.py
```

```python
import os
from typing import TypedDict

import mlflow
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class ServiceState(TypedDict):
    query: str
    answer: str


def handle_query(state: ServiceState) -> dict:
    env = os.environ.get("APP_ENV", "local")
    return {"answer": f"[{env}] Processed query: {state['query']}"}


builder = StateGraph(ServiceState)
builder.add_node("handle_query", handle_query)
builder.add_edge(START, "handle_query")
builder.add_edge("handle_query", END)
graph = builder.compile()

app = FastAPI(title="Dockerized LangGraph Service")


class InvokeRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/invoke")
def invoke(req: InvokeRequest):
    return graph.invoke({"query": req.query, "answer": ""})
```

## Conclusion
Packaging your compiled graph behind FastAPI + Uvicorn in a slim Python Docker image gives you portable, reproducible deployments anywhere containers run.
