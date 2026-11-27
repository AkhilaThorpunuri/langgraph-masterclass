"""Integrating with CrewAI and Other Frameworks -- LangGraph as the outer orchestrator.

Objective:
    Show how to wrap an external multi-agent framework (CrewAI Crew / AutoGen)
    inside a normal LangGraph node so LangGraph owns state, routing, and tracing.
"""
import socket
from typing import TypedDict

import mlflow
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


# STEP 1 -- EXTERNAL CREW STAND-IN (swap for `crewai.Crew(...).kickoff(...)`).
class MockExternalCrew:
    def kickoff(self, inputs: dict) -> str:
        return f"[External Crew Report] Deep-dive analysis for '{inputs['topic']}'."


# STEP 2 -- STATE.
class OrchestratorState(TypedDict):
    topic: str
    crew_report: str
    final_brief: str


# STEP 3 -- NODES (LangGraph calls the external crew inside a regular node).
def run_external_crew_node(state: OrchestratorState) -> dict:
    # state coming IN:  {"topic": "AI Agent Frameworks", "crew_report": "", "final_brief": ""}
    # state going OUT:  {"crew_report": "[External Crew Report] ..."}
    report = MockExternalCrew().kickoff(inputs={"topic": state["topic"]})
    return {"crew_report": report}


def postprocess_brief(state: OrchestratorState) -> dict:
    # state coming IN:  {"crew_report": "...", ...}
    # state going OUT:  {"final_brief": "Executive Summary -> ..."}
    return {"final_brief": f"Executive Summary -> {state['crew_report']}"}


# STEP 4 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(OrchestratorState)
builder.add_node("external_crew", run_external_crew_node)
builder.add_node("postprocess_brief", postprocess_brief)
builder.add_edge(START, "external_crew")
builder.add_edge("external_crew", "postprocess_brief")
builder.add_edge("postprocess_brief", END)
graph = builder.compile()

# STEP 5 -- INVOKE:
print("\n--- LangGraph + External Crew Orchestration ---")
out = graph.invoke({"topic": "AI Agent Frameworks", "crew_report": "", "final_brief": ""})
print(f"  Topic:       {out['topic']}")
print(f"  Crew Output: {out['crew_report']}")
print(f"  Final Brief: {out['final_brief']}")
