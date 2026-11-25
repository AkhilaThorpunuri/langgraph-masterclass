"""Long-Term Memory (Store) -- cross-thread memory with InMemoryStore.

Demonstrates:
- BaseStore / InMemoryStore vs checkpointer (thread-scoped vs user-scoped memory)
- Storing and reading user preferences across two DIFFERENT thread_ids
- Searching namespaced memories inside a node using the `store` argument
"""
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

store = InMemoryStore()
graph = builder.compile(checkpointer=MemorySaver(), store=store)

# Thread 1: save a user memory
t1 = {"configurable": {"thread_id": "thread-1", "user_id": "user-42"}}
print(graph.invoke({"user_input": "remember: Prefers concise Python examples", "reply": ""}, t1))

# Thread 2 (brand new thread!): still recalls user-42's long-term memory from the Store
t2 = {"configurable": {"thread_id": "thread-2", "user_id": "user-42"}}
print(graph.invoke({"user_input": "What do you know about me?", "reply": ""}, t2))
