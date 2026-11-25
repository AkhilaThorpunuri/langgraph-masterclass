"""LangGraph Studio -- stepping through execution and editing state mid-run.

Demonstrates:
- How LangGraph Studio inspects and steps through a compiled graph using checkpoints
- Simulating a step-by-step Studio session with interrupt_before + update_state
"""
from typing import TypedDict

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class StudioState(TypedDict):
    query: str
    plan: str
    result: str


def plan_node(state: StudioState) -> dict:
    return {"plan": f"Step 1: research {state['query']}"}


def execute_node(state: StudioState) -> dict:
    return {"result": f"Executed -> {state['plan']}"}


builder = StateGraph(StudioState)
builder.add_node("plan_node", plan_node)
builder.add_node("execute_node", execute_node)
builder.add_edge(START, "plan_node")
builder.add_edge("plan_node", "execute_node")
builder.add_edge("execute_node", END)

# Export `graph` so `langgraph dev` / LangGraph Studio can load it directly
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["execute_node"])

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "studio-step-1"}}
    graph.invoke({"query": "observability", "plan": "", "result": ""}, cfg)
    snap = graph.get_state(cfg)
    print("Paused before:", snap.next, "| Current plan:", snap.values["plan"])

    # Edit state mid-run (just like clicking Edit State in LangGraph Studio)
    graph.update_state(cfg, {"plan": "Step 1: research observability + MLflow tracing"})
    final = graph.invoke(None, cfg)
    print("Resumed final state:", final)
