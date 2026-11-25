# Multi-Agent Research Assistant
> Venkata Bhattaram (c) 2026

## Contents
* Planner-researcher-writer architecture
* Dynamic parallel research with Send()
* Final report synthesis

## Overview
A multi-agent research assistant combines **planning**, **dynamic map-reduce parallelism**, and **report synthesis**:
1. **`planner`** breaks a broad topic into a dynamic list of focused subtopics.
2. **`dispatch_researchers`** emits one `Send("researcher", {"subtopic": ...})` per subtopic so every researcher runs concurrently in the same superstep.
3. **`writer`** waits for all parallel researchers to finish, reads the aggregated `findings` list (merged via `operator.add`), and synthesizes a structured Markdown report.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 14-real-world-projects/code/multi_agent_research_assistant.py
```

```python
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
    return {"subtopics": [f"{topic}: architecture", f"{topic}: observability", f"{topic}: production trade-offs"]}


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

print(graph.invoke({"topic": "Stateful AI Agents", "subtopics": [], "findings": [], "report": ""})["report"])
```

## Conclusion
Combining a `planner` node with `Send()` fan-out and a `writer` fan-in node scales effortlessly to any number of research subtopics—and MLflow traces visualize all parallel researcher spans side by side.
