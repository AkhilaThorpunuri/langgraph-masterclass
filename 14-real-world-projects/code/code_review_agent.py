"""Code Review Agent -- reading a diff, running parallel reviewers, and posting comments.

Demonstrates:
- Parsing a git diff into changed lines
- Fanning out in parallel to security_review and style_review nodes (same superstep)
- Fanning in to compile_review to produce structured PR comments
"""
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
    added = [
        line[1:].strip()
        for line in state["diff"].splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return {"added_lines": added}


def security_review(state: ReviewState) -> dict:
    findings = []
    for line in state["added_lines"]:
        if "eval(" in line or "password =" in line.lower():
            findings.append(f"[SECURITY] Risky pattern detected: `{line}`")
    return {"comments": findings or ["[SECURITY] No obvious vulnerabilities found."]}


def style_review(state: ReviewState) -> dict:
    findings = []
    for line in state["added_lines"]:
        if "print(" in line:
            findings.append(f"[STYLE] Prefer structured logging over print(): `{line}`")
    return {"comments": findings or ["[STYLE] Clean formatting."]}


def compile_review(state: ReviewState) -> dict:
    bullet_list = "\n".join(f" - {c}" for c in state["comments"])
    return {"summary": f"PR Review ({len(state['comments'])} comment(s)):\n{bullet_list}"}


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

SAMPLE_DIFF = """+++ b/app.py
+def handle(user_input):
+    print("debugging:", user_input)
+    return eval(user_input)
"""

res = graph.invoke({"diff": SAMPLE_DIFF, "added_lines": [], "comments": [], "summary": ""})
print(res["summary"])
