# Code Review Agent
> Venkata Bhattaram (c) 2026

## Contents
* Reading a diff
* Parallel multi-perspective review (security + style)
* Compiling structured PR comments

## Overview
An effective automated code reviewer runs multiple specialized checks in parallel rather than stuffing every rule into one monolithic prompt:
1. **`parse_diff`** extracts added lines from a unified git diff.
2. **`security_review`** and **`style_review`** fan out in the **same superstep**, each appending findings into `comments: Annotated[list[str], operator.add]`.
3. **`compile_review`** fans in once both reviewers finish and formats the combined PR review summary.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 14-real-world-projects/code/code_review_agent.py
```

```python
import operator
from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class ReviewState(TypedDict):
    diff: str
    added_lines: list[str]
    comments: Annotated[list[str], operator.add]
    summary: str


def parse_diff(state: ReviewState) -> dict:
    return {"added_lines": [l[1:].strip() for l in state["diff"].splitlines() if l.startswith("+") and not l.startswith("+++")]}


def security_review(state: ReviewState) -> dict:
    findings = [f"[SECURITY] Risky pattern: `{l}`" for l in state["added_lines"] if "eval(" in l or "password =" in l.lower()]
    return {"comments": findings or ["[SECURITY] No obvious vulnerabilities found."]}


def style_review(state: ReviewState) -> dict:
    findings = [f"[STYLE] Prefer logging over print(): `{l}`" for l in state["added_lines"] if "print(" in l]
    return {"comments": findings or ["[STYLE] Clean formatting."]}


def compile_review(state: ReviewState) -> dict:
    return {"summary": "PR Review:\n" + "\n".join(f" - {c}" for c in state["comments"])}


builder = StateGraph(ReviewState)
builder.add_node("parse_diff", parse_diff)
builder.add_node("security_review", security_review)
builder.add_node("style_review", style_review)
builder.add_node("compile_review", compile_review)
builder.add_edge(START, "parse_diff")
builder.add_edge("parse_diff", "security_review")
builder.add_edge("parse_diff", "style_review")
builder.add_edge("security_review", "compile_review")
builder.add_edge("style_review", "compile_review")
builder.add_edge("compile_review", END)
graph = builder.compile()

SAMPLE_DIFF = "+++ b/app.py\n+def handle(x):\n+    print(x)\n+    return eval(x)\n"
print(graph.invoke({"diff": SAMPLE_DIFF, "added_lines": [], "comments": [], "summary": ""})["summary"])
```

## Conclusion
Fan-out/fan-in lets you add new reviewer perspectives (performance, tests, docs) as independent nodes without slowing down the pipeline or complicating existing reviewers.
