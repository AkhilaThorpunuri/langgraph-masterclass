"""Basic RAG Graph -- Retrieve node -> Generate node over a lightweight knowledge base.

Demonstrates:
- Wiring a retrieval step and a grounded generation step into a StateGraph
- Using simple keyword-overlap scoring (swap for FAISS / vector store in production)
"""
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()

KNOWLEDGE_BASE = [
    "LangGraph represents agent workflows as stateful graphs with nodes and edges.",
    "Checkpointers save graph state after every superstep, keyed by thread_id.",
    "Conditional edges inspect state and choose the next node dynamically at runtime.",
    "ToolNode executes tool_calls emitted by a chat model and returns ToolMessages.",
]


class RAGState(TypedDict):
    question: str
    docs: list[str]
    answer: str


def retrieve(state: RAGState) -> dict:
    terms = set(state["question"].lower().split())
    scored = sorted(
        KNOWLEDGE_BASE,
        key=lambda doc: len(terms & set(doc.lower().split())),
        reverse=True,
    )
    return {"docs": scored[:2]}


def generate(state: RAGState) -> dict:
    context = " ".join(state["docs"])
    return {"answer": f"Based on retrieved docs: {context}"}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

res = graph.invoke({"question": "How do checkpointers save state?", "docs": [], "answer": ""})
print("Retrieved docs:", res["docs"])
print("Answer:", res["answer"])
