"""Streaming to a Frontend -- FastAPI Server-Sent Events (SSE) endpoint for a LangGraph app.

Objective:
    Show how to expose graph.stream(..., stream_mode="updates") as an HTTP
    text/event-stream endpoint so a browser UI displays each node as it finishes.
"""
import json
import socket
from typing import TypedDict

import mlflow
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import END, START, StateGraph


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
class JobState(TypedDict):
    topic: str
    outline: str
    article: str


# STEP 2 -- NODES.
def make_outline(state: JobState) -> dict:
    # state coming IN:  {"topic": "LangGraph", "outline": "", "article": ""}
    # state going OUT:  {"outline": "1. Intro to LangGraph\n2. Key Patterns"}
    return {"outline": f"1. Intro to {state['topic']}\n2. Key Patterns"}


def write_article(state: JobState) -> dict:
    # state coming IN:  {"outline": "..."}
    # state going OUT:  {"article": "Full article built from:\n..."}
    return {"article": f"Full article built from:\n{state['outline']}"}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(JobState)
builder.add_node("make_outline", make_outline)
builder.add_node("write_article", write_article)
builder.add_edge(START, "make_outline")
builder.add_edge("make_outline", "write_article")
builder.add_edge("write_article", END)
graph = builder.compile()

# STEP 4 -- FASTAPI SSE ENDPOINT:
# Serve live with: uvicorn streaming_to_a_frontend:app --reload
app = FastAPI(title="LangGraph SSE Streaming Demo")


@app.get("/stream")
def stream_graph(topic: str = "LangGraph"):
    def event_generator():
        for update in graph.stream({"topic": topic, "outline": "", "article": ""}, stream_mode="updates"):
            yield f"data: {json.dumps(update)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# STEP 5 -- LOCAL SMOKE TEST (prints exact SSE frames without needing a browser):
if __name__ == "__main__":
    print("\n--- Streaming to Frontend (SSE Events) ---")
    for update in graph.stream({"topic": "LangGraph", "outline": "", "article": ""}, stream_mode="updates"):
        for node, payload in update.items():
            print(f"  SSE event [{node:>13}] -> {json.dumps(payload)}")
    print("  SSE event [         done] -> [DONE]")
