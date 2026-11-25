"""Unit Testing Nodes and Graphs -- testing nodes in isolation and full graph runs.

Demonstrates:
- Testing pure node functions directly without compiling the graph
- Injecting a fake/mock LLM callable so graph tests stay deterministic and fast
- Runnable with either `python unit_testing_nodes_and_graphs.py` or `pytest`
"""
from typing import Callable, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class ReviewState(TypedDict):
    text: str
    sentiment: str
    reply: str


def classify_node(state: ReviewState, llm_fn: Callable[[str], str]) -> dict:
    return {"sentiment": llm_fn(state["text"])}


def reply_node(state: ReviewState) -> dict:
    if state["sentiment"] == "negative":
        return {"reply": "We're sorry to hear that -- escalating to support."}
    return {"reply": "Thank you for the positive feedback!"}


def build_graph(llm_fn: Callable[[str], str]):
    builder = StateGraph(ReviewState)
    builder.add_node("classify", lambda s: classify_node(s, llm_fn))
    builder.add_node("reply", reply_node)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "reply")
    builder.add_edge("reply", END)
    return builder.compile()


# --- Unit Tests ---
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
    test_reply_node_in_isolation()
    test_full_graph_with_mock_llm()
    print("All unit tests passed!")
