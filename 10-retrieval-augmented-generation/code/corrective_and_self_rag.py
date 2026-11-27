"""Corrective and Self-RAG -- grading retrieved docs, rewriting the query, and looping.

Objective:
    Show how a self-correcting RAG graph grades retrieved docs and loops back
    through `rewrite_query` -> `retrieve` when the first search misses.
"""
import socket
from typing import TypedDict

import mlflow
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

DOCS = [
    "SqliteSaver persists LangGraph checkpoints to a local SQLite database file.",
    "InMemoryStore stores cross-thread long-term memories by namespace.",
]


# STEP 1 -- STATE.
class CorrectiveRAGState(TypedDict):
    query: str
    docs: list[str]
    relevant: bool
    retries: int
    answer: str


# STEP 2 -- NODES.
def retrieve(state: CorrectiveRAGState) -> dict:
    # Pass 1 IN:  query="how to save stuff?"                -> OUT: docs=[]
    # Pass 2 IN:  query="SqliteSaver checkpoints database"  -> OUT: docs=["SqliteSaver ..."]
    q = state["query"].lower()
    hits = [d for d in DOCS if any(word in d.lower() for word in q.split() if len(word) > 3)]
    return {"docs": hits}


def grade_docs(state: CorrectiveRAGState) -> dict:
    # state going OUT: {"relevant": True/False}
    return {"relevant": len(state["docs"]) > 0}


def rewrite_query(state: CorrectiveRAGState) -> dict:
    # Reformulates vague query into domain terms and increments retry counter.
    return {"query": "SqliteSaver checkpoints database", "retries": state.get("retries", 0) + 1}


def generate(state: CorrectiveRAGState) -> dict:
    if state["docs"]:
        return {"answer": f"Grounded answer: {state['docs'][0]}"}
    return {"answer": "Could not find relevant documents after retries."}


# STEP 3 -- ROUTING FUNCTION: proceed to generate if relevant or retry cap hit; else rewrite.
def route_after_grade(state: CorrectiveRAGState) -> str:
    if state["relevant"] or state.get("retries", 0) >= 2:
        return "generate"
    return "rewrite_query"


# STEP 4 -- GRAPH, EDGES & COMPILE.
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

# STEP 5 -- INVOKE: watch pass 1 fail grading -> rewrite_query -> pass 2 succeed!
print("\n--- Corrective / Self-RAG Demo ---")
res = graph.invoke({"query": "how to save stuff?", "docs": [], "relevant": False, "retries": 0, "answer": ""})
print("  Original query:  how to save stuff?")
print(f"  Rewritten query: {res['query']} (after {res['retries']} retry)")
print(f"  Answer:          {res['answer']}")
