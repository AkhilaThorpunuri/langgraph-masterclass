"""Checkpointers -- persisting graph state across invocations by thread_id.

Demonstrates:
- MemorySaver (InMemorySaver) vs SqliteSaver (when langgraph-checkpoint-sqlite is installed)
- Scoped persistence using config={"configurable": {"thread_id": ...}}
- Inspecting latest state with get_state() and full history with get_state_history()
"""
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class CounterState(TypedDict):
    count: int


def increment(state: CounterState) -> dict:
    return {"count": state.get("count", 0) + 1}


builder = StateGraph(CounterState)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo-thread-1"}}

for run_num in (1, 2, 3):
    result = graph.invoke({}, config)
    print(f"Run {run_num} -> count={result['count']}")

latest = graph.get_state(config)
print("Latest checkpoint_id:", latest.config["configurable"]["checkpoint_id"])

history = list(graph.get_state_history(config))
print(f"Total checkpoints recorded for thread: {len(history)}")
