"""Dockerizing a Graph -- FastAPI HTTP wrapper + Dockerfile template for container deploys.

Demonstrates:
- Exposing a compiled LangGraph app over HTTP with FastAPI (`POST /invoke`, `GET /health`)
- Reading runtime configuration from environment variables
- Printing a production-ready Dockerfile when run directly
"""
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


DOCKERFILE_TEMPLATE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "dockerizing_a_graph:app", "--host", "0.0.0.0", "--port", "8000"]
"""

if __name__ == "__main__":
    print("=== Dockerfile Template ===")
    print(DOCKERFILE_TEMPLATE)
    print("=== Local Smoke Test ===")
    print(invoke(InvokeRequest(query="container readiness check")))
