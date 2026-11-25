# Hierarchical Agent Teams
> Venkata Bhattaram (c) 2026

## Contents
* Teams as subgraphs
* Nested supervisors
* Scaling to many agents

## Overview
When a single supervisor has to manage too many workers, you can group related workers into **team subgraphs**—for example, a `research_team` (`web_searcher -> fact_checker`) and a `writing_team` (`drafter -> editor`). Each team is compiled as its own independent `StateGraph` and added directly as a node inside a top-level supervisor graph.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 08-multi-agent-systems/code/hierarchical_agent_teams.py
```

```python
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


def route_teams(state: SharedTeamState) -> str:
    if state.get("stage", "init") == "init":
        return "research_team"
    if state.get("stage") == "researched":
        return "writing_team"
    return END


root = StateGraph(SharedTeamState)
root.add_node("top_supervisor", lambda s: {})
root.add_node("research_team", research_team)
root.add_node("writing_team", writing_team)
root.add_edge(START, "top_supervisor")
root.add_conditional_edges("top_supervisor", route_teams, {"research_team": "research_team", "writing_team": "writing_team", END: END})
root.add_edge("research_team", "top_supervisor")
root.add_edge("writing_team", "top_supervisor")
graph = root.compile()

print(graph.invoke({"topic": "Hierarchical Multi-Agent Systems", "artifacts": [], "stage": "init"})["artifacts"])
```

## Conclusion
Nesting compiled subgraphs lets teams evolve, be tested, and be traced independently while a top-level supervisor coordinates the overall workflow.
