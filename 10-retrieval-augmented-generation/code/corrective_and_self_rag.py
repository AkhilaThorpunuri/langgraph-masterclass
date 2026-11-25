"""Corrective and Self-RAG -- grading retrieved docs, rewriting the query, and looping.

Demonstrates:
- Grading retrieved documents for relevance
- Rewriting an ambiguous query and re-retrieving when relevance is low
- Stopping cleanly after a maximum number of correction loops
"""
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()

DOCS = [
    "SqliteSaver persists LangGraph checkpoints to a local SQLite database file.",
    "InMemoryStore stores cross-thread long-term memories by namespace.",
]


class CorrectiveRAGState(TypedDict):
    query: str
    docs: list[str]
    relevant: bool
    retries: int
    answer: str


def retrieve(state: CorrectiveRAGState) -> dict:
    q = state["query"].lower()
    hits = [d for d in DOCS if any(word in d.lower() for word in q.split() if len(word) > 3)]
    return {"docs": hits}


def grade_docs(state: CorrectiveRAGState) -> dict:
    return {"relevant": len(state["docs"]) > 0}


def rewrite_query(state: CorrectiveRAGState) -> dict:
    # Expand vague user phrasing into domain terms
    return {
        "query": "SqliteSaver checkpoints database",
        "retries": state.get("retries", 0) + 1,
    }


def generate(state: CorrectiveRAGState) -> dict:
    if state["docs"]:
        return {"answer": f"Grounded answer: {state['docs'][0]}"}
    return {"answer": "Could not find relevant documents after retries."}


def route_after_grade(state: CorrectiveRAGState) -> str:
    if state["relevant"] or state.get("retries", 0) >= 2:
        return "generate"
    return "rewrite_query"


builder = StateGraph(CorrectiveRAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_docs", grade_docs)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_docs")
builder.add_conditional_edges(
    "grade_docs",
    route_after_grade,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate", END)
graph = builder.compile()

res = graph.invoke({"query": "how to save stuff?", "docs": [], "relevant": False, "retries": 0, "answer": ""})
print("Final query:", res["query"], "| Retries:", res["retries"])
print("Answer:", res["answer"])
