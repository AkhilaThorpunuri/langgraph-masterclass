# Basic RAG Graph
> Venkata Bhattaram (c) 2026

## Contents
* Retrieve node
* Generate node
* Wiring a knowledge base into a graph

## Overview
**Retrieval-Augmented Generation (RAG)** grounds an LLM's answer in external documents. Modeling RAG as a LangGraph `StateGraph` (`START -> retrieve -> generate -> END`) makes retrieval explicit in state (`docs: list[str]`), easy to inspect in MLflow traces, and ready to extend later with document grading, query rewriting, or adaptive routing.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 10-retrieval-augmented-generation/code/basic_rag_graph.py
```

```python
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
    scored = sorted(KNOWLEDGE_BASE, key=lambda doc: len(terms & set(doc.lower().split())), reverse=True)
    return {"docs": scored[:2]}


def generate(state: RAGState) -> dict:
    return {"answer": f"Based on retrieved docs: {' '.join(state['docs'])}"}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

print(graph.invoke({"question": "How do checkpointers save state?", "docs": [], "answer": ""}))
```

## Conclusion
Representing RAG as separate `retrieve` and `generate` nodes keeps your retrieval context inspectable and lays the foundation for Corrective RAG and Adaptive RAG.
