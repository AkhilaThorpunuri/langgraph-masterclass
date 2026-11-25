"""Structured Output -- Pydantic response models, validation, and retry loop.

Demonstrates:
- Validating model output against a Pydantic BaseModel schema
- Looping back to retry when validation fails (up to max_retries)
"""
from typing import Literal, Optional, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


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


def generate_candidate(state: StructuredState) -> dict:
    attempt = state.get("attempt", 0) + 1
    # Simulate an invalid first attempt (priority=99) that self-corrects on attempt 2
    if attempt == 1:
        candidate = {"category": "billing", "priority": 99, "summary": "Double charge"}
    else:
        candidate = {"category": "billing", "priority": 4, "summary": "Customer charged twice for order"}
    return {"attempt": attempt, "raw_output": candidate}


def validate_output(state: StructuredState) -> dict:
    try:
        model = TicketClassification.model_validate(state["raw_output"])
        return {"parsed": model.model_dump(), "error": ""}
    except ValidationError as exc:
        return {"parsed": None, "error": str(exc.errors()[0]["msg"])}


def route_after_validation(state: StructuredState) -> str:
    if state.get("parsed") is not None or state.get("attempt", 0) >= 3:
        return END
    return "generate_candidate"


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

res = graph.invoke({"raw_ticket": "Charged twice!", "attempt": 0, "raw_output": {}, "parsed": None, "error": ""})
print("Attempts taken:", res["attempt"])
print("Validated structured output:", res["parsed"])
