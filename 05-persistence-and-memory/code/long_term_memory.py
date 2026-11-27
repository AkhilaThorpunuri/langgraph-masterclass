"""Long-Term Memory (Store) -- cross-thread memory with InMemoryStore.

Objective:
    Contrast thread-scoped checkpointers with user-scoped BaseStore memory:
    save a preference in thread-1 and recall it inside a brand-new thread-2.
"""
import socket
from typing import TypedDict

import mlflow
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


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


# STEP 1 -- STATE: per-turn user input and assistant reply.
class ChatState(TypedDict):
    user_input: str
    reply: str


# STEP 2 -- NODES: `*, store: BaseStore` is auto-injected by LangGraph at runtime.
def remember_and_reply(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    text = state["user_input"]
    if text.lower().startswith("remember:"):
        # state coming IN:  {"user_input": "remember: ...", "reply": ""}
        # side effect:      store.put(("memories", user_id), "fact-N", {"fact": ...})
        # state going OUT:  {"reply": "Saved to long-term memory: ..."}
        fact = text.split(":", 1)[1].strip()
        existing = store.search(namespace)
        store.put(namespace, f"fact-{len(existing) + 1}", {"fact": fact})
        return {"reply": f"Saved to long-term memory: {fact}"}

    # state coming IN:  {"user_input": "What do you know about me?", "reply": ""}
    # state going OUT:  {"reply": "Known memories for user-42: [...]"}
    memories = [item.value["fact"] for item in store.search(namespace)]
    return {"reply": f"Known memories for {user_id}: {memories}"}


# STEP 3 -- GRAPH, EDGES & COMPILE WITH BOTH CHECKPOINTER AND STORE.
builder = StateGraph(ChatState)
builder.add_node("remember_and_reply", remember_and_reply)
builder.add_edge(START, "remember_and_reply")
builder.add_edge("remember_and_reply", END)

store = InMemoryStore()
graph = builder.compile(checkpointer=MemorySaver(), store=store)

# STEP 4 -- INVOKE ACROSS TWO DIFFERENT THREADS FOR THE SAME USER:
print("\n--- Long-Term Memory Demo (user-42) ---")
t1 = {"configurable": {"thread_id": "thread-1", "user_id": "user-42"}}
r1 = graph.invoke({"user_input": "remember: Prefers concise Python examples", "reply": ""}, t1)
print(f"  [Thread 1] User:  {r1['user_input']}")
print(f"  [Thread 1] Reply: {r1['reply']}")

t2 = {"configurable": {"thread_id": "thread-2", "user_id": "user-42"}}
r2 = graph.invoke({"user_input": "What do you know about me?", "reply": ""}, t2)
print(f"\n  [Thread 2] User:  {r2['user_input']}")
print(f"  [Thread 2] Reply: {r2['reply']}")
