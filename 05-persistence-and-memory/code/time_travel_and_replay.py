"""Time Travel and Replay -- inspecting checkpoint history and forking a past run.

Objective:
    Show how to walk back to an earlier checkpoint, edit state at that exact
    point with update_state(), and replay only the downstream nodes from there.
"""
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


# STEP 1 -- STATE: tracks topic -> draft -> polished text across two nodes.
class PipelineState(TypedDict):
    topic: str
    draft: str
    polished: str


# STEP 2 -- NODES.
def write_draft(state: PipelineState) -> dict:
    # state coming IN:  {"topic": "LangGraph", "draft": "", "polished": ""}
    # state going OUT: {"draft": "Initial draft about LangGraph"}
    return {"draft": f"Initial draft about {state['topic']}"}


def polish_draft(state: PipelineState) -> dict:
    # state coming IN:  {"draft": "..."}
    # state going OUT: {"polished": "[POLISHED] ..."}
    return {"polished": f"[POLISHED] {state['draft']}"}


# STEP 3 -- GRAPH, EDGES & COMPILE WITH CHECKPOINTER.
builder = StateGraph(PipelineState)
builder.add_node("write_draft", write_draft)
builder.add_node("polish_draft", polish_draft)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "polish_draft")
builder.add_edge("polish_draft", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "time-travel-demo"}}

# STEP 4 -- ORIGINAL RUN: executes both write_draft and polish_draft.
print("\n--- 1. Original Run ---")
original = graph.invoke({"topic": "LangGraph", "draft": "", "polished": ""}, config)
print(f"  Draft:    {original['draft']}")
print(f"  Polished: {original['polished']}")

# STEP 5 -- TIME TRAVEL: locate the checkpoint right before polish_draft ran.
history = list(graph.get_state_history(config))
print(f"\n--- 2. Saved Checkpoints ({len(history)} total) ---")
for idx, snap in enumerate(history, start=1):
    next_step = snap.next[0] if snap.next else "DONE"
    print(f"  Snapshot {idx} | next: {next_step:<12} | draft: {snap.values.get('draft', '')!r}")

before_polish = next(s for s in history if s.next == ("polish_draft",))

# STEP 6 -- FORK & REPLAY: edit `draft` at that past checkpoint and resume with None.
fork_config = graph.update_state(
    before_polish.config,
    {"draft": "Forked draft about LangGraph Time Travel"},
)
replayed = graph.invoke(None, fork_config)
print("\n--- 3. Forked & Replayed Run ---")
print(f"  Edited Draft: {replayed['draft']}")
print(f"  New Polished: {replayed['polished']}")
