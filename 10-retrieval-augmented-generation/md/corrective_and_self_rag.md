# Corrective and Self-RAG
> Venkata Bhattaram (c) 2026

## Contents
* Grading retrieved documents
* Query rewriting
* Self-correction loops

## Overview
Basic RAG blindly passes whatever the retriever returned into the generator—even when the retrieved documents are irrelevant. **Corrective RAG** inserts a `grade_docs` node after `retrieve`: if the documents are not relevant, a conditional edge routes to `rewrite_query` to reformulate the user's question and loops back to `retrieve` (capped by a max retry count so the loop always terminates).

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 10-retrieval-augmented-generation/code/corrective_and_self_rag.py
```

```python
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
    return {"docs": [d for d in DOCS if any(w in d.lower() for w in q.split() if len(w) > 3)]}


def grade_docs(state: CorrectiveRAGState) -> dict:
    return {"relevant": len(state["docs"]) > 0}


def rewrite_query(state: CorrectiveRAGState) -> dict:
    return {"query": "SqliteSaver checkpoints database", "retries": state.get("retries", 0) + 1}


def generate(state: CorrectiveRAGState) -> dict:
    return {"answer": f"Grounded answer: {state['docs'][0]}" if state["docs"] else "No relevant docs found."}


def route_after_grade(state: CorrectiveRAGState) -> str:
    return "generate" if state["relevant"] or state.get("retries", 0) >= 2 else "rewrite_query"


builder = StateGraph(CorrectiveRAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_docs", grade_docs)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_docs")
builder.add_conditional_edges("grade_docs", route_after_grade, {"generate": "generate", "rewrite_query": "rewrite_query"})
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate", END)
graph = builder.compile()

print(graph.invoke({"query": "how to save stuff?", "docs": [], "relevant": False, "retries": 0, "answer": ""}))
```

## Conclusion
Adding a grading and query-rewriting cycle dramatically reduces hallucinations by ensuring only relevant context reaches the generation step.
