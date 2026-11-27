"""Multi-Agent Research Assistant -- Planner -> Parallel Send() Researchers -> Writer.

Objective:
    Show a dynamic map-reduce multi-agent pipeline: `planner` generates subtopics,
    `dispatch_researchers` spawns one parallel Send("researcher", ...) branch per
    subtopic, and `writer` aggregates all findings into a final Markdown report.
"""
import operator
import socket
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


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


# STEP 1 -- PARENT & WORKER STATES.
class ResearchReportState(TypedDict):
    topic: str
    subtopics: list[str]
    findings: Annotated[list[str], operator.add]
    report: str


class SubtopicState(TypedDict):
    subtopic: str


# STEP 2 -- NODES & DYNAMIC SEND DISPATCHER.
def planner(state: ResearchReportState) -> dict:
    # Superstep 1: breaks the broad topic into 3 focused subtopics.
    topic = state["topic"]
    return {
        "subtopics": [
            f"{topic}: architecture",
            f"{topic}: observability",
            f"{topic}: production trade-offs",
        ]
    }


def dispatch_researchers(state: ResearchReportState) -> list[Send]:
    # Map step: spawns N parallel researcher branches (one per subtopic).
    return [Send("researcher", {"subtopic": st}) for st in state["subtopics"]]


def researcher(state: SubtopicState) -> dict:
    # Superstep 2: each parallel branch receives only its own SubtopicState and appends to `findings`.
    return {"findings": [f"Finding for '{state['subtopic']}': verified key best practices."]}


def writer(state: ResearchReportState) -> dict:
    # Superstep 3 (Reduce step): waits for all researchers and synthesizes the final report.
    sections = "\n".join(f"- {f}" for f in state["findings"])
    return {"report": f"# Research Report: {state['topic']}\n\n{sections}"}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(ResearchReportState)
builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", dispatch_researchers, ["researcher"])
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)
graph = builder.compile()

# STEP 4 -- INVOKE:
print("\n--- Multi-Agent Research Assistant Demo ---")
out = graph.invoke({"topic": "Stateful AI Agents", "subtopics": [], "findings": [], "report": ""})
print(out["report"])
