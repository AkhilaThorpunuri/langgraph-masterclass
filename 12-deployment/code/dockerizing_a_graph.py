"""Dockerizing a Graph -- FastAPI HTTP wrapper + Dockerfile template for container deploys.

Objective:
    Show how to wrap a compiled LangGraph app in FastAPI (`/health` + `/invoke`)
    and package it cleanly in a slim Python Docker container.
"""
import os
import socket
from typing import TypedDict

import mlflow
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if _is_port_open("127.0.0.1", 5000):
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("masterclass")
    mlflow.langchain.autolog()
else:
    print("[MLflow] Server not running at http://localhost:5000/ -- skipping remote tracing.")


# STEP 1 -- STATE.
class ServiceState(TypedDict):
    query: str
    answer: str


# STEP 2 -- NODES (reads runtime config from environment variables, never hardcoded secrets).
def handle_query(state: ServiceState) -> dict:
    # state coming IN:  {"query": "container readiness check", "answer": ""}
    # state going OUT:  {"answer": "[local] Processed query: container readiness check"}
    env = os.environ.get("APP_ENV", "local")
    return {"answer": f"[{env}] Processed query: {state['query']}"}


# STEP 3 -- COMPILE GRAPH.
builder = StateGraph(ServiceState)
builder.add_node("handle_query", handle_query)
builder.add_edge(START, "handle_query")
builder.add_edge("handle_query", END)
graph = builder.compile()

# STEP 4 -- FASTAPI SERVICE WRAPPER.
app = FastAPI(title="Dockerized LangGraph Service")


class InvokeRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/invoke")
def invoke(req: InvokeRequest):
    return graph.invoke({"query": req.query, "answer": ""})


# STEP 5 -- DOCKERFILE TEMPLATE & LOCAL SMOKE TEST.
DOCKERFILE_TEMPLATE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "dockerizing_a_graph:app", "--host", "0.0.0.0", "--port", "8000"]
"""

if __name__ == "__main__":
    print("\n--- Dockerfile Template ---")
    print(DOCKERFILE_TEMPLATE.strip())
    res = invoke(InvokeRequest(query="container readiness check"))
    print("\n--- Local Smoke Test ---")
    print(f"  Query:  {res['query']}")
    print(f"  Answer: {res['answer']}")
