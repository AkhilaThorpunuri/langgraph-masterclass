# Unit Testing Nodes and Graphs
> Venkata Bhattaram (c) 2026

## Contents
* Testing nodes in isolation
* Mocking LLM calls
* Testing full graph runs

## Overview
Because LangGraph nodes are plain Python functions (`state -> partial_state_dict`), testing them is straightforward:
1. **Node unit tests**: call the node function directly with a sample dict and assert on its returned update—no graph compilation or LLM needed.
2. **Graph integration tests**: inject a deterministic fake/mock LLM callable into `build_graph(llm_fn)` so full graph runs execute in milliseconds with predictable outputs, ideal for CI with `pytest`.

## Code Example
Run directly with Python or with `pytest`:
```powershell
python 11-testing-debugging-and-observability/code/unit_testing_nodes_and_graphs.py
pytest 11-testing-debugging-and-observability/code/unit_testing_nodes_and_graphs.py
```

```python
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


def test_reply_node_in_isolation():
    assert "escalating" in reply_node({"text": "", "sentiment": "negative", "reply": ""})["reply"]


def test_full_graph_with_mock_llm():
    mock_llm = lambda text: "negative" if "broken" in text.lower() else "positive"
    out = build_graph(mock_llm).invoke({"text": "The widget arrived broken", "sentiment": "", "reply": ""})
    assert out["sentiment"] == "negative"
```

## Conclusion
Dependency-injecting your LLM client lets you verify routing, reducers, and edge cases in CI without flaky network calls or API costs.
