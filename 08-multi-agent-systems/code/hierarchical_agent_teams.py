"""Hierarchical Agent Teams -- nested team subgraphs under a top-level supervisor.

Objective:
    Show how two independent team subgraphs (research_team and writing_team)
    are compiled separately and added as nodes inside a parent supervisor graph.
"""
import operator
import socket
from typing import Annotated, TypedDict

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


# STEP 1 -- SHARED STATE: used by both team subgraphs and the top-level supervisor.
class SharedTeamState(TypedDict):
    topic: str
    artifacts: Annotated[list[str], operator.add]
    stage: str


# STEP 2 -- TEAM 1 SUBGRAPH (research_team: web_searcher -> fact_checker).
def web_searcher(state: SharedTeamState) -> dict:
    return {"artifacts": [f"[Research/Search] Sources found for {state['topic']}"]}


def fact_checker(state: SharedTeamState) -> dict:
    # Marks stage="researched" so the top supervisor knows to hand off to writing_team.
    return {"artifacts": ["[Research/FactCheck] Verified 3 primary citations"], "stage": "researched"}


research_builder = StateGraph(SharedTeamState)
research_builder.add_node("web_searcher", web_searcher)
research_builder.add_node("fact_checker", fact_checker)
research_builder.add_edge(START, "web_searcher")
research_builder.add_edge("web_searcher", "fact_checker")
research_builder.add_edge("fact_checker", END)
research_team = research_builder.compile()


# STEP 3 -- TEAM 2 SUBGRAPH (writing_team: drafter -> editor).
def drafter(state: SharedTeamState) -> dict:
    return {"artifacts": [f"[Writing/Drafter] Drafted article on {state['topic']}"]}


def editor(state: SharedTeamState) -> dict:
    # Marks stage="done" so the top supervisor exits to END.
    return {"artifacts": ["[Writing/Editor] Finalized formatting"], "stage": "done"}


writing_builder = StateGraph(SharedTeamState)
writing_builder.add_node("drafter", drafter)
writing_builder.add_node("editor", editor)
writing_builder.add_edge(START, "drafter")
writing_builder.add_edge("drafter", "editor")
writing_builder.add_edge("editor", END)
writing_team = writing_builder.compile()


# STEP 4 -- TOP-LEVEL SUPERVISOR GRAPH: adds compiled teams directly as nodes.
def top_supervisor(state: SharedTeamState) -> dict:
    return {}


def route_teams(state: SharedTeamState) -> str:
    # stage="init" -> research_team | stage="researched" -> writing_team | stage="done" -> END
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

# STEP 5 -- INVOKE: runs research_team subgraph, then writing_team subgraph, then END.
print("\n--- Hierarchical Agent Teams Demo ---")
out = graph.invoke({"topic": "Hierarchical Multi-Agent Systems", "artifacts": [], "stage": "init"})
for idx, line in enumerate(out["artifacts"], start=1):
    print(f"  Step {idx}: {line}")
print(f"  Final stage: {out['stage']}")
