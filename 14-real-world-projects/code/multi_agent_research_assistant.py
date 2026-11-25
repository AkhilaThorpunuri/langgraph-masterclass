"""Multi-Agent Research Assistant -- Planner -> Parallel Send() Researchers -> Writer.

Demonstrates:
- Planner node decomposing a research question into subtopics
- Dynamic parallel fan-out using Send("researcher", ...) for each subtopic
- Aggregating findings via operator.add and synthesizing a final markdown report
"""
import operator
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class ResearchReportState(TypedDict):
    topic: str
    subtopics: list[str]
    findings: Annotated[list[str], operator.add]
    report: str


class SubtopicState(TypedDict):
    subtopic: str


def planner(state: ResearchReportState) -> dict:
    topic = state["topic"]
    return {
        "subtopics": [
            f"{topic}: architecture",
            f"{topic}: observability",
            f"{topic}: production trade-offs",
        ]
    }


def dispatch_researchers(state: ResearchReportState) -> list[Send]:
    return [Send("researcher", {"subtopic": st}) for st in state["subtopics"]]


def researcher(state: SubtopicState) -> dict:
    return {"findings": [f"Finding for '{state['subtopic']}': verified key best practices."]}


def writer(state: ResearchReportState) -> dict:
    sections = "\n".join(f"- {f}" for f in state["findings"])
    return {"report": f"# Research Report: {state['topic']}\n\n{sections}"}


builder = StateGraph(ResearchReportState)
builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", dispatch_researchers, ["researcher"])
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)
graph = builder.compile()

out = graph.invoke({"topic": "Stateful AI Agents", "subtopics": [], "findings": [], "report": ""})
print(out["report"])
