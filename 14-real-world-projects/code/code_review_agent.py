"""Code Review Agent -- reading a diff, running parallel reviewers, and posting comments.

Objective:
    Show a real-world fan-out/fan-in pipeline: parse_diff fans out to parallel
    security_review and style_review nodes (same superstep), then compile_review
    fans in and formats the combined PR comments.
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


# STEP 1 -- STATE: `comments` uses operator.add so parallel reviewers append safely.
class ReviewState(TypedDict):
    diff: str
    added_lines: list[str]
    comments: Annotated[list[str], operator.add]
    summary: str


# STEP 2 -- NODES.
def parse_diff(state: ReviewState) -> dict:
    # Superstep 1: extracts newly added lines from the unified git diff.
    added = [
        line[1:].strip()
        for line in state["diff"].splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return {"added_lines": added}


def security_review(state: ReviewState) -> dict:
    # Superstep 2 (parallel with style_review): checks for risky calls/secrets.
    findings = [
        f"[SECURITY] Risky pattern detected: `{line}`"
        for line in state["added_lines"]
        if "eval(" in line or "password =" in line.lower()
    ]
    return {"comments": findings or ["[SECURITY] No obvious vulnerabilities found."]}


def style_review(state: ReviewState) -> dict:
    # Superstep 2 (parallel with security_review): checks for debug prints.
    findings = [
        f"[STYLE] Prefer structured logging over print(): `{line}`"
        for line in state["added_lines"]
        if "print(" in line
    ]
    return {"comments": findings or ["[STYLE] Clean formatting."]}


def compile_review(state: ReviewState) -> dict:
    # Superstep 3 (fan-in): waits for both reviewers, then formats final PR summary.
    bullet_list = "\n".join(f" - {c}" for c in state["comments"])
    return {"summary": f"PR Review ({len(state['comments'])} comment(s)):\n{bullet_list}"}


# STEP 3 -- GRAPH, EDGES & COMPILE.
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

# STEP 4 -- INVOKE WITH SAMPLE DIFF:
SAMPLE_DIFF = """+++ b/app.py
+def handle(user_input):
+    print("debugging:", user_input)
+    return eval(user_input)
"""

print("\n--- Code Review Agent Demo ---")
res = graph.invoke({"diff": SAMPLE_DIFF, "added_lines": [], "comments": [], "summary": ""})
print(res["summary"])
