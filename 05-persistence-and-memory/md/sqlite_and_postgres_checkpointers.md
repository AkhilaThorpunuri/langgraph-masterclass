# SQLite and Postgres Checkpointers
> Venkata Bhattaram (c) 2026

## Contents
* SqliteSaver
* PostgresSaver
* Choosing a backend for production

## Overview
While `MemorySaver` (`InMemorySaver`) is convenient for quick experiments and unit tests, its state disappears as soon as the Python process exits. Real applications need durable checkpoint storage so conversations, workflows, and human-in-the-loop pauses survive restarts.

LangGraph exposes the exact same checkpointer interface across storage backends:
* **`SqliteSaver`** (`langgraph-checkpoint-sqlite`): stores checkpoints in a single local `.db` file using Python's built-in SQLite engine. Ideal for local development, CLI tools, and single-process apps.
* **`PostgresSaver`** (`langgraph-checkpoint-postgres`): stores checkpoints in PostgreSQL for multi-worker, horizontally scaled production services where concurrent writers share state safely.

Because the graph code only depends on `builder.compile(checkpointer=checkpointer)`, swapping from `MemorySaver` to `SqliteSaver` to `PostgresSaver` requires changing only a few lines at startup—zero changes to your nodes or edges.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 05-persistence-and-memory/code/sqlite_and_postgres_checkpointers.py
```

```python
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

ctx = SqliteSaver.from_conn_string(":memory:") if HAS_SQLITE_SAVER else nullcontext(MemorySaver())

with ctx as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    thread_a = {"configurable": {"thread_id": "user-alice"}}
    thread_b = {"configurable": {"thread_id": "user-bob"}}

    print("Alice run 1:", graph.invoke({}, thread_a))
    print("Alice run 2:", graph.invoke({}, thread_a))
    print("Bob run 1:  ", graph.invoke({}, thread_b))
```

## Conclusion
Start locally with `SqliteSaver` (or `MemorySaver` in tests) and move to `PostgresSaver` when you run multiple workers or containers. Each `thread_id` stays completely isolated inside the checkpointer, so `user-alice` and `user-bob` never collide.
