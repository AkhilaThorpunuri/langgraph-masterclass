"""Building a Custom Agent Loop -- agent node + tool node + memory + guardrails.

Objective:
    Show why you build a custom StateGraph agent: insert a guardrail node before
    the agent so unsafe requests short-circuit straight to END without running tools.
"""
import socket

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


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

BLOCKED_KEYWORDS = ("drop table", "delete database", "rm -rf")


# STEP 1 -- TOOL & CUSTOM STATE (extends MessagesState with a `blocked` flag).
@tool
def lookup_order(order_id: str) -> str:
    """Look up an order status by ID."""
    return f"Order {order_id} is SHIPPED (ETA: tomorrow)."


class GuardedAgentState(MessagesState):
    blocked: bool


# STEP 2 -- NODES.
def guardrail(state: GuardedAgentState) -> dict:
    # state coming IN:  {"messages": [..., HumanMessage(...)], "blocked": False}
    # state going OUT:  {"blocked": True, "messages": [AIMessage("Blocked...")]} OR {"blocked": False}
    last_text = state["messages"][-1].content.lower()
    if any(bad in last_text for bad in BLOCKED_KEYWORDS):
        return {
            "blocked": True,
            "messages": [AIMessage(content="Blocked by guardrail: unsafe request detected.")],
        }
    return {"blocked": False}


def route_after_guardrail(state: GuardedAgentState) -> str:
    # Short-circuit directly to END if blocked; otherwise proceed to agent.
    return END if state.get("blocked") else "agent"


def agent_node(state: GuardedAgentState) -> dict:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"Here is your update: {last.content}")]}
    return {
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "ord_1", "name": "lookup_order", "args": {"order_id": "ORD-1042"}}])
        ]
    }


# STEP 3 -- GRAPH, EDGES & COMPILE WITH CHECKPOINTER.
builder = StateGraph(GuardedAgentState)
builder.add_node("guardrail", guardrail)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode([lookup_order]))

builder.add_edge(START, "guardrail")
builder.add_conditional_edges("guardrail", route_after_guardrail, {"agent": "agent", END: END})
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "custom-agent-1"}}

# STEP 4 -- INVOKE SAFE vs UNSAFE REQUESTS:
print("\n--- Custom Agent Loop with Guardrail ---")
r1 = graph.invoke({"messages": [HumanMessage(content="Check status of ORD-1042")], "blocked": False}, cfg)
print(f"  Safe request   -> {r1['messages'][-1].content}")

r2 = graph.invoke({"messages": [HumanMessage(content="Please run DROP TABLE users")], "blocked": False}, cfg)
print(f"  Unsafe request -> {r2['messages'][-1].content}")
