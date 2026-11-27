"""SQLite and Postgres Checkpointers -- durable persistence backends & thread isolation.

Objective:
    Show how SqliteSaver / PostgresSaver plug into the exact same compile() API
    as MemorySaver, and how separate thread_ids keep each user's state isolated.
"""
from contextlib import nullcontext
import socket
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
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

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_SQLITE_SAVER = True
except ImportError:
    SqliteSaver = None
    HAS_SQLITE_SAVER = False


# STEP 1 -- STATE: list of notes persisted per thread_id.
class NoteState(TypedDict):
    notes: list[str]


# STEP 2 -- NODES: appends the next numbered note to the thread's existing list.
def add_note(state: NoteState) -> dict:
    # state coming IN:  {"notes": [...]}  (empty on first run for a thread)
    # state going OUT: {"notes": [..., "note-K"]}
    existing = list(state.get("notes", []))
    return {"notes": existing + [f"note-{len(existing) + 1}"]}


# STEP 3 -- GRAPH & EDGES.
builder = StateGraph(NoteState)
builder.add_node("add_note", add_note)
builder.add_edge(START, "add_note")
builder.add_edge("add_note", END)

# STEP 4 -- COMPILE WITH BACKEND:
# Uses SqliteSaver if installed, otherwise falls back cleanly to MemorySaver.
# For production multi-worker services, swap in PostgresSaver behind the same API.
ctx = SqliteSaver.from_conn_string(":memory:") if HAS_SQLITE_SAVER else nullcontext(MemorySaver())

with ctx as checkpointer:
    backend = "SqliteSaver" if HAS_SQLITE_SAVER else "MemorySaver (fallback)"
    print(f"\n--- Checkpointer Backend: {backend} ---")
    graph = builder.compile(checkpointer=checkpointer)

    # STEP 5 -- INVOKE ACROSS TWO ISOLATED THREADS:
    thread_a = {"configurable": {"thread_id": "user-alice"}}
    thread_b = {"configurable": {"thread_id": "user-bob"}}

    print("  Alice (run 1) ->", graph.invoke({}, thread_a)["notes"])
    print("  Alice (run 2) ->", graph.invoke({}, thread_a)["notes"])
    print("  Bob   (run 1) ->", graph.invoke({}, thread_b)["notes"])
