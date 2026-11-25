"""Hierarchical Agent Teams -- nested team subgraphs under a top-level supervisor.

Demonstrates:
- Two independent team subgraphs (research_team and writing_team)
- A top-level supervisor orchestrating both teams as compiled subgraph nodes
"""
import operator
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class SharedTeamState(TypedDict):
    topic: str
    artifacts: Annotated[list[str], operator.add]
    stage: str


# --- Team 1: Research Subgraph ---
def web_searcher(state: SharedTeamState) -> dict:
    return {"artifacts": [f"[Research/Search] Sources found for {state['topic']}"]}


def fact_checker(state: SharedTeamState) -> dict:
    return {"artifacts": ["[Research/FactCheck] Verified 3 primary citations"], "stage": "researched"}


research_builder = StateGraph(SharedTeamState)
research_builder.add_node("web_searcher", web_searcher)
research_builder.add_node("fact_checker", fact_checker)
research_builder.add_edge(START, "web_searcher")
research_builder.add_edge("web_searcher", "fact_checker")
research_builder.add_edge("fact_checker", END)
research_team = research_builder.compile()


# --- Team 2: Writing Subgraph ---
def drafter(state: SharedTeamState) -> dict:
    return {"artifacts": [f"[Writing/Drafter] Drafted article on {state['topic']}"]}


def editor(state: SharedTeamState) -> dict:
    return {"artifacts": ["[Writing/Editor] Finalized formatting"], "stage": "done"}


writing_builder = StateGraph(SharedTeamState)
writing_builder.add_node("drafter", drafter)
writing_builder.add_node("editor", editor)
writing_builder.add_edge(START, "drafter")
writing_builder.add_edge("drafter", "editor")
writing_builder.add_edge("editor", END)
writing_team = writing_builder.compile()


# --- Top-Level Supervisor ---
def top_supervisor(state: SharedTeamState) -> dict:
    return {}


def route_teams(state: SharedTeamState) -> str:
    stage = state.get("stage", "init")
    if stage == "init":
        return "research_team"
    if stage == "researched":
        return "writing_team"
    return END


root = StateGraph(SharedTeamState)
root.add_node("top_supervisor", top_supervisor)
root.add_node("research_team", research_team)
root.add_node("writing_team", writing_team)
root.add_edge(START, "top_supervisor")
root.add_conditional_edges(
    "top_supervisor",
    route_teams,
    {"research_team": "research_team", "writing_team": "writing_team", END: END},
)
root.add_edge("research_team", "top_supervisor")
root.add_edge("writing_team", "top_supervisor")
graph = root.compile()

out = graph.invoke({"topic": "Hierarchical Multi-Agent Systems", "artifacts": [], "stage": "init"})
for line in out["artifacts"]:
    print(line)
