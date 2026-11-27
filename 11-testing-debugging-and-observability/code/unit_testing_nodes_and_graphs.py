"""Unit Testing Nodes and Graphs -- testing nodes in isolation and full graph runs.

Objective:
    Show how to unit-test pure node functions directly AND test full compiled
    graphs deterministically by injecting a mock LLM callable (pytest-ready).
"""
import socket
from typing import Callable, TypedDict

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


# STEP 1 -- STATE.
class ReviewState(TypedDict):
    text: str
    sentiment: str
    reply: str


# STEP 2 -- NODES (classify_node accepts an injected llm_fn for easy mocking).
def classify_node(state: ReviewState, llm_fn: Callable[[str], str]) -> dict:
    # state coming IN:  {"text": "The widget arrived broken", ...}
    # state going OUT:  {"sentiment": "negative"}
    return {"sentiment": llm_fn(state["text"])}


def reply_node(state: ReviewState) -> dict:
    # Pure function: depends ONLY on state["sentiment"], so trivial to unit-test in isolation.
    if state["sentiment"] == "negative":
        return {"reply": "We're sorry to hear that -- escalating to support."}
    return {"reply": "Thank you for the positive feedback!"}


# STEP 3 -- GRAPH FACTORY (inject real LLM in prod, mock_llm in tests).
def build_graph(llm_fn: Callable[[str], str]):
    builder = StateGraph(ReviewState)
    builder.add_node("classify", lambda s: classify_node(s, llm_fn))
    builder.add_node("reply", reply_node)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "reply")
    builder.add_edge("reply", END)
    return builder.compile()


# STEP 4 -- UNIT TESTS (run via `python unit_testing_nodes_and_graphs.py` or `pytest`).
def test_reply_node_in_isolation():
    assert "escalating" in reply_node({"text": "", "sentiment": "negative", "reply": ""})["reply"]
    assert "Thank you" in reply_node({"text": "", "sentiment": "positive", "reply": ""})["reply"]


def test_full_graph_with_mock_llm():
    mock_llm = lambda text: "negative" if "broken" in text.lower() else "positive"
    graph = build_graph(mock_llm)
    out = graph.invoke({"text": "The widget arrived broken", "sentiment": "", "reply": ""})
    assert out["sentiment"] == "negative"
    assert "escalating" in out["reply"]


if __name__ == "__main__":
    print("\n--- Unit Testing Nodes & Graphs ---")
    test_reply_node_in_isolation()
    print("  [PASS] test_reply_node_in_isolation")
    test_full_graph_with_mock_llm()
    print("  [PASS] test_full_graph_with_mock_llm")
