"""SQLite and Postgres Checkpointers -- durable persistence backends.

Demonstrates:
- Using SqliteSaver (falls back to MemorySaver if langgraph-checkpoint-sqlite is not installed)
- Showing the drop-in PostgresSaver pattern for production multi-process deployments
- Verifying thread isolation across two distinct thread_ids
"""
from contextlib import nullcontext
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()

try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    HAS_SQLITE_SAVER = True
except ImportError:
    SqliteSaver = None
    HAS_SQLITE_SAVER = False


class NoteState(TypedDict):
    notes: list[str]


def add_note(state: NoteState) -> dict:
    existing = list(state.get("notes", []))
    return {"notes": existing + [f"note-{len(existing) + 1}"]}


builder = StateGraph(NoteState)
builder.add_node("add_note", add_note)
builder.add_edge(START, "add_note")
builder.add_edge("add_note", END)

# Production Postgres pattern (requires langgraph-checkpoint-postgres + psycopg):
#   from langgraph.checkpoint.postgres import PostgresSaver
#   with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as checkpointer:
#       checkpointer.setup()
#       graph = builder.compile(checkpointer=checkpointer)

ctx = (
    SqliteSaver.from_conn_string(":memory:")
    if HAS_SQLITE_SAVER
    else nullcontext(MemorySaver())
)

with ctx as checkpointer:
    backend = "SqliteSaver" if HAS_SQLITE_SAVER else "MemorySaver (fallback)"
    print(f"Using checkpointer backend: {backend}")
    graph = builder.compile(checkpointer=checkpointer)

    thread_a = {"configurable": {"thread_id": "user-alice"}}
    thread_b = {"configurable": {"thread_id": "user-bob"}}

    print("Alice run 1:", graph.invoke({}, thread_a))
    print("Alice run 2:", graph.invoke({}, thread_a))
    print("Bob run 1:  ", graph.invoke({}, thread_b))
