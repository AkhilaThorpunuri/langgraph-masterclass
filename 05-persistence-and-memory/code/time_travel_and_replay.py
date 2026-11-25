"""Time Travel and Replay -- inspecting checkpoint history and forking a past run.

Demonstrates:
- Running a multi-step graph with a checkpointer
- Listing all historical checkpoints with get_state_history()
- Updating state at a past checkpoint (update_state) and forking execution from it
"""
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class PipelineState(TypedDict):
    topic: str
    draft: str
    polished: str


def write_draft(state: PipelineState) -> dict:
    return {"draft": f"Initial draft about {state['topic']}"}


def polish_draft(state: PipelineState) -> dict:
    return {"polished": f"[POLISHED] {state['draft']}"}


builder = StateGraph(PipelineState)
builder.add_node("write_draft", write_draft)
builder.add_node("polish_draft", polish_draft)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "polish_draft")
builder.add_edge("polish_draft", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "time-travel-demo"}}

original = graph.invoke({"topic": "LangGraph", "draft": "", "polished": ""}, config)
print("Original run:", original)

history = list(graph.get_state_history(config))
print(f"\nCheckpoint history ({len(history)} snapshots, newest first):")
for snap in history:
    print(" ", snap.config["configurable"]["checkpoint_id"], "next=", snap.next, "values=", snap.values)

# Pick the checkpoint right after write_draft (where next == ('polish_draft',))
before_polish = next(s for s in history if s.next == ("polish_draft",))

# Fork from that past checkpoint with an edited draft
fork_config = graph.update_state(
    before_polish.config,
    {"draft": "Forked draft about LangGraph Time Travel"},
)
replayed = graph.invoke(None, fork_config)
print("\nForked & replayed run:", replayed)
