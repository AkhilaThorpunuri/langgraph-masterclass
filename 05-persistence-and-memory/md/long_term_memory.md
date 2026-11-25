# Long-Term Memory (Store)
> Venkata Bhattaram (c) 2026

## Contents
* BaseStore vs checkpointer
* Cross-thread memory
* Namespaced memory search

## Overview
A **checkpointer** is scoped to a single `thread_id` (short-term conversational or workflow state). When an agent needs to remember user preferences, facts, or profile details across *separate* conversations or threads, LangGraph provides **`BaseStore`** (with `InMemoryStore` for local dev and `PostgresStore` for production).

Compile your graph with `store=store`, and any node that declares `*, store: BaseStore` in its signature automatically receives the store instance. Items are organized by tuple **namespaces** such as `("memories", user_id)`, allowing clean isolation per user and fast retrieval via `store.search(namespace)`.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 05-persistence-and-memory/code/long_term_memory.py
```

```python
from typing import TypedDict

import mlflow
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class ChatState(TypedDict):
    user_input: str
    reply: str


def remember_and_reply(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    text = state["user_input"]
    if text.lower().startswith("remember:"):
        fact = text.split(":", 1)[1].strip()
        existing = store.search(namespace)
        store.put(namespace, f"fact-{len(existing) + 1}", {"fact": fact})
        return {"reply": f"Saved to long-term memory: {fact}"}

    memories = [item.value["fact"] for item in store.search(namespace)]
    return {"reply": f"Known memories for {user_id}: {memories}"}


builder = StateGraph(ChatState)
builder.add_node("remember_and_reply", remember_and_reply)
builder.add_edge(START, "remember_and_reply")
builder.add_edge("remember_and_reply", END)

graph = builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())

t1 = {"configurable": {"thread_id": "thread-1", "user_id": "user-42"}}
print(graph.invoke({"user_input": "remember: Prefers concise Python examples", "reply": ""}, t1))

t2 = {"configurable": {"thread_id": "thread-2", "user_id": "user-42"}}
print(graph.invoke({"user_input": "What do you know about me?", "reply": ""}, t2))
```

## Conclusion
Use a **checkpointer** for within-thread execution history and **`BaseStore`** for cross-thread user or application knowledge. Together, they give agents both short-term working memory and durable long-term recall.
