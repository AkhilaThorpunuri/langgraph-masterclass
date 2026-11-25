"""Integrating with CrewAI and Other Frameworks -- LangGraph as the outer orchestrator.

Demonstrates:
- Wrapping an external multi-agent framework (e.g. a CrewAI Crew or AutoGen team)
  inside a standard LangGraph node so LangGraph owns routing, state, checkpoints, and tracing
"""
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class MockExternalCrew:
    """Stand-in for a CrewAI Crew.kickoff(inputs=...) call so this demo runs offline."""

    def kickoff(self, inputs: dict) -> str:
        return f"[External Crew Report] Deep-dive analysis for '{inputs['topic']}'."


class OrchestratorState(TypedDict):
    topic: str
    crew_report: str
    final_brief: str


def run_external_crew_node(state: OrchestratorState) -> dict:
    crew = MockExternalCrew()
    report = crew.kickoff(inputs={"topic": state["topic"]})
    return {"crew_report": report}


def postprocess_brief(state: OrchestratorState) -> dict:
    return {"final_brief": f"Executive Summary -> {state['crew_report']}"}


builder = StateGraph(OrchestratorState)
builder.add_node("external_crew", run_external_crew_node)
builder.add_node("postprocess_brief", postprocess_brief)
builder.add_edge(START, "external_crew")
builder.add_edge("external_crew", "postprocess_brief")
builder.add_edge("postprocess_brief", END)
graph = builder.compile()

out = graph.invoke({"topic": "AI Agent Frameworks", "crew_report": "", "final_brief": ""})
print(out["final_brief"])
