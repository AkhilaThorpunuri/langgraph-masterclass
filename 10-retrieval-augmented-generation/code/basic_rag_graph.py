"""Basic RAG Graph -- Retrieve node -> Generate node over a lightweight knowledge base.

Objective:
    Show the foundational two-node RAG graph (START -> retrieve -> generate -> END)
    keeping retrieved documents explicit in state for easy debugging and tracing.
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

KNOWLEDGE_BASE = [
    "LangGraph represents agent workflows as stateful graphs with nodes and edges.",
    "Checkpointers save graph state after every superstep, keyed by thread_id.",
    "Conditional edges inspect state and choose the next node dynamically at runtime.",
    "ToolNode executes tool_calls emitted by a chat model and returns ToolMessages.",
]


# STEP 1 -- STATE.
class RAGState(TypedDict):
    question: str
    docs: list[str]
    answer: str


# STEP 2 -- NODES.
def retrieve(state: RAGState) -> dict:
    # state coming IN:  {"question": "How do checkpointers save state?", "docs": [], "answer": ""}
    # state going OUT:  {"docs": [<top-2 matching documents>]}
    terms = set(state["question"].lower().split())
    scored = sorted(
        KNOWLEDGE_BASE,
        key=lambda doc: len(terms & set(doc.lower().split())),
        reverse=True,
    )
    return {"docs": scored[:2]}


def generate(state: RAGState) -> dict:
    # state coming IN:  {"docs": [...], ...}
    # state going OUT:  {"answer": "Based on retrieved docs: ..."}
    context = " ".join(state["docs"])
    return {"answer": f"Based on retrieved docs: {context}"}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# STEP 4 -- INVOKE:
print("\n--- Basic RAG Graph Demo ---")
res = graph.invoke({"question": "How do checkpointers save state?", "docs": [], "answer": ""})
print(f"  Question: {res['question']}")
print("  Retrieved docs:")
for idx, doc in enumerate(res["docs"], start=1):
    print(f"    {idx}. {doc}")
print(f"  Answer:   {res['answer']}")
