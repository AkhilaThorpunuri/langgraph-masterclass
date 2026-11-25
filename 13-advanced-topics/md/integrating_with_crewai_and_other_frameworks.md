# Integrating with CrewAI and Other Frameworks
> Venkata Bhattaram (c) 2026

## Contents
* LangGraph as an orchestrator around CrewAI crews
* Comparing LangGraph, CrewAI, and AutoGen
* When to combine frameworks

## Overview
You do not have to choose exclusively between LangGraph and role-based multi-agent frameworks like **CrewAI** or **AutoGen**. Because any Python callable can be a LangGraph node, you can wrap a CrewAI `Crew.kickoff(...)` call inside a single node while letting **LangGraph** own the outer workflow—state schema, deterministic routing, checkpoints, human-in-the-loop interrupts, and MLflow tracing.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 13-advanced-topics/code/integrating_with_crewai_and_other_frameworks.py
```

```python
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class MockExternalCrew:
    def kickoff(self, inputs: dict) -> str:
        return f"[External Crew Report] Deep-dive analysis for '{inputs['topic']}'."


class OrchestratorState(TypedDict):
    topic: str
    crew_report: str
    final_brief: str


def run_external_crew_node(state: OrchestratorState) -> dict:
    return {"crew_report": MockExternalCrew().kickoff(inputs={"topic": state["topic"]})}


def postprocess_brief(state: OrchestratorState) -> dict:
    return {"final_brief": f"Executive Summary -> {state['crew_report']}"}


builder = StateGraph(OrchestratorState)
builder.add_node("external_crew", run_external_crew_node)
builder.add_node("postprocess_brief", postprocess_brief)
builder.add_edge(START, "external_crew")
builder.add_edge("external_crew", "postprocess_brief")
builder.add_edge("postprocess_brief", END)
graph = builder.compile()

print(graph.invoke({"topic": "AI Agent Frameworks", "crew_report": "", "final_brief": ""})["final_brief"])
```

## Conclusion
Wrapping external agent crews inside LangGraph nodes gives you the best of both worlds: high-level role-based collaboration inside the node and strict, stateful orchestration around it.
