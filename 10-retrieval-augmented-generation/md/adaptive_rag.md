# Adaptive RAG
> Venkata Bhattaram (c) 2026

## Contents
* Routing between retrieval strategies
* Falling back to web search
* Combining sources

## Overview
Not every user question needs the same retrieval pipeline. **Adaptive RAG** starts with a routing function right at `START` that classifies the question and picks the best strategy:
* **Vectorstore**: for internal/domain documentation questions.
* **Web search**: for fresh or real-time information ("latest", "today", "news").
* **Direct answer**: for greetings or general knowledge that require no retrieval at all.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 10-retrieval-augmented-generation/code/adaptive_rag.py
```

```python
from typing import Literal, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class AdaptiveRAGState(TypedDict):
    question: str
    strategy: str
    context: str
    answer: str


def route_question(state: AdaptiveRAGState) -> Literal["vectorstore", "web_search", "direct_answer"]:
    q = state["question"].lower()
    if any(k in q for k in ("langgraph", "checkpoint", "reducer")):
        return "vectorstore"
    if any(k in q for k in ("today", "latest", "news", "weather")):
        return "web_search"
    return "direct_answer"


def vectorstore_node(state: AdaptiveRAGState) -> dict:
    return {"strategy": "vectorstore", "context": "Internal docs: LangGraph uses Annotated reducers to merge parallel updates."}


def web_search_node(state: AdaptiveRAGState) -> dict:
    return {"strategy": "web_search", "context": "Live web snippet: Latest LangGraph release adds enhanced streaming support."}


def direct_answer_node(state: AdaptiveRAGState) -> dict:
    return {"strategy": "direct_answer", "context": "No external retrieval needed.", "answer": f"Direct answer for: {state['question']}"}


def synthesize(state: AdaptiveRAGState) -> dict:
    return {"answer": f"[{state['strategy']}] {state['context']}"}


builder = StateGraph(AdaptiveRAGState)
builder.add_node("vectorstore", vectorstore_node)
builder.add_node("web_search", web_search_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("synthesize", synthesize)
builder.add_conditional_edges(START, route_question, {"vectorstore": "vectorstore", "web_search": "web_search", "direct_answer": "direct_answer"})
builder.add_edge("vectorstore", "synthesize")
builder.add_edge("web_search", "synthesize")
builder.add_edge("synthesize", END)
builder.add_edge("direct_answer", END)
graph = builder.compile()

for q in ["How does a LangGraph reducer work?", "What is the latest news on AI agents today?", "Say hello!"]:
    print(graph.invoke({"question": q, "strategy": "", "context": "", "answer": ""})["answer"])
```

## Conclusion
Adaptive RAG lowers latency and cost by avoiding unnecessary vector searches on trivial prompts while dynamically routing fresh queries to web search.
