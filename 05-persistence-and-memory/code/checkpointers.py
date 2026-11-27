"""Checkpointers -- persisting graph state across invocations by thread_id.

Objective:
    Show how a checkpointer saves graph state after every superstep, keyed by
    `thread_id`, so repeated `invoke()` calls pick up where the last one left off.
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


# STEP 1 -- STATE: shared object persisted by the checkpointer after every superstep.
class CounterState(TypedDict):
    count: int


# STEP 2 -- NODES: reads the last checkpointed count (or 0 on run 1) and increments it.
def increment(state: CounterState) -> dict:
    # state coming IN:  {"count": N}  (empty {} on run 1 -> defaults to 0)
    # state going OUT: {"count": N + 1}
    return {"count": state.get("count", 0) + 1}


# STEP 3 -- GRAPH & EDGES: linear graph START -> increment -> END.
builder = StateGraph(CounterState)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

# STEP 4 -- COMPILE WITH CHECKPOINTER: MemorySaver records every step in RAM.
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# STEP 5 -- INVOKE: thread_id scopes persistence to one conversation/session.
config = {"configurable": {"thread_id": "demo-thread-1"}}

# Run 1 IN: {} -> OUT: {'count': 1}
# Run 2 IN: {} (restores count=1 from checkpoint) -> OUT: {'count': 2}
# Run 3 IN: {} (restores count=2 from checkpoint) -> OUT: {'count': 3}
print("\n--- Checkpointers Demo (thread: demo-thread-1) ---")
for run_num in (1, 2, 3):
    result = graph.invoke({}, config)
    print(f"  Run {run_num} -> count = {result['count']}")

# STEP 6 -- INSPECT HISTORY: inspect latest checkpoint and full thread timeline.
latest = graph.get_state(config)
history = list(graph.get_state_history(config))
print(f"\n  Latest count saved:  {latest.values['count']}")
print(f"  Checkpoints stored:  {len(history)}")
