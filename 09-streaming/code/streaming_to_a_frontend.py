"""Streaming to a Frontend -- FastAPI Server-Sent Events (SSE) endpoint for a LangGraph app.

Demonstrates:
- Wrapping graph.stream(..., stream_mode="updates") in a FastAPI StreamingResponse (SSE)
- Running an in-process test client when executed directly so no server setup is required
  (run with `uvicorn streaming_to_a_frontend:app --reload` to serve live HTTP SSE)
"""
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


if __name__ == "__main__":
    print("Simulating SSE events emitted by GET /stream?topic=LangGraph:")
    for update in graph.stream({"topic": "LangGraph", "outline": "", "article": ""}, stream_mode="updates"):
        print(f"data: {json.dumps(update)}\n")
    print("data: [DONE]")
