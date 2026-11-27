"""Structured Output -- Pydantic response models, validation, and retry loop.

Objective:
    Show how to validate model JSON against a Pydantic schema and loop back
    to retry when validation fails (bounded by max retries).
"""
import socket
from typing import Literal, Optional, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError


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


# STEP 1 -- PYDANTIC CONTRACT & GRAPH STATE.
class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "general"]
    priority: int = Field(ge=1, le=5)
    summary: str = Field(min_length=5)


class StructuredState(TypedDict):
    raw_ticket: str
    attempt: int
    raw_output: dict
    parsed: Optional[dict]
    error: str


# STEP 2 -- NODES.
def generate_candidate(state: StructuredState) -> dict:
    # Attempt 1 produces invalid priority=99 -> triggers validation retry.
    # Attempt 2 produces valid priority=4   -> passes Pydantic validation.
    attempt = state.get("attempt", 0) + 1
    if attempt == 1:
        candidate = {"category": "billing", "priority": 99, "summary": "Double charge"}
    else:
        candidate = {"category": "billing", "priority": 4, "summary": "Customer charged twice for order"}
    return {"attempt": attempt, "raw_output": candidate}


def validate_output(state: StructuredState) -> dict:
    # state coming IN:  {"raw_output": {...}, "attempt": N}
    # state going OUT:  {"parsed": dict, "error": ""} OR {"parsed": None, "error": "..."}
    try:
        model = TicketClassification.model_validate(state["raw_output"])
        return {"parsed": model.model_dump(), "error": ""}
    except ValidationError as exc:
        return {"parsed": None, "error": str(exc.errors()[0]["msg"])}


# STEP 3 -- ROUTING FUNCTION: stop if valid or after 3 attempts, else retry.
def route_after_validation(state: StructuredState) -> str:
    if state.get("parsed") is not None or state.get("attempt", 0) >= 3:
        return END
    return "generate_candidate"


# STEP 4 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(StructuredState)
builder.add_node("generate_candidate", generate_candidate)
builder.add_node("validate_output", validate_output)
builder.add_edge(START, "generate_candidate")
builder.add_edge("generate_candidate", "validate_output")
builder.add_conditional_edges(
    "validate_output",
    route_after_validation,
    {"generate_candidate": "generate_candidate", END: END},
)
graph = builder.compile()

# STEP 5 -- INVOKE: runs 2 attempts and exits with a validated TicketClassification dict.
print("\n--- Structured Output with Validation Retry ---")
res = graph.invoke({"raw_ticket": "Charged twice!", "attempt": 0, "raw_output": {}, "parsed": None, "error": ""})
print(f"  Attempts taken: {res['attempt']}")
print(f"  Category:       {res['parsed']['category']}")
print(f"  Priority:       {res['parsed']['priority']}")
print(f"  Summary:        {res['parsed']['summary']}")
