"""LangGraph Platform and Server -- structuring a graph export and langgraph.json config.

Demonstrates:
- Exporting a module-level `graph` object (the contract expected by `langgraph.json`)
- Printing a valid `langgraph.json` manifest pointing at this module
"""
import json
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class EchoState(TypedDict):
    message: str
    reply: str


def respond(state: EchoState) -> dict:
    return {"reply": f"LangGraph Server received: {state['message']}"}


builder = StateGraph(EchoState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

# Exported symbol referenced by langgraph.json -> "graphs": {"echo": "./langgraph_platform_and_server.py:graph"}
graph = builder.compile()

LANGGRAPH_JSON_EXAMPLE = {
    "dependencies": ["."],
    "graphs": {"echo": "./langgraph_platform_and_server.py:graph"},
    "env": ".env",
}

if __name__ == "__main__":
    print("Sample langgraph.json:")
    print(json.dumps(LANGGRAPH_JSON_EXAMPLE, indent=2))
    print("\nLocal invoke test:")
    print(graph.invoke({"message": "Hello Platform!", "reply": ""}))
