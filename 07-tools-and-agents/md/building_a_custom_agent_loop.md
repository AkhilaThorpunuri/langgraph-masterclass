# Building a Custom Agent Loop
> Venkata Bhattaram (c) 2026

## Contents
* Agent node + tool node + conditional edge
* Adding memory
* Adding guardrails

## Overview
When you build a custom `StateGraph` instead of using a prebuilt agent, you control every transition in the agent loop. A common production pattern adds an explicit **guardrail node** right after `START`: if the incoming user request violates policy, the graph short-circuits straight to `END` without ever calling the model or any tool. Safe requests proceed into the `agent ↔ ToolNode` cycle, backed by a checkpointer for multi-turn memory.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 07-tools-and-agents/code/building_a_custom_agent_loop.py
```

```python
import mlflow
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()

BLOCKED_KEYWORDS = ("drop table", "delete database", "rm -rf")


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order status by ID."""
    return f"Order {order_id} is SHIPPED (ETA: tomorrow)."


class GuardedAgentState(MessagesState):
    blocked: bool


def guardrail(state: GuardedAgentState) -> dict:
    last_text = state["messages"][-1].content.lower()
    if any(bad in last_text for bad in BLOCKED_KEYWORDS):
        return {"blocked": True, "messages": [AIMessage(content="Blocked by guardrail: unsafe request detected.")]}
    return {"blocked": False}


def route_after_guardrail(state: GuardedAgentState) -> str:
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
print(graph.invoke({"messages": [HumanMessage(content="Check status of ORD-1042")], "blocked": False}, cfg)["messages"][-1].content)
print(graph.invoke({"messages": [HumanMessage(content="Please run DROP TABLE users")], "blocked": False}, cfg)["messages"][-1].content)
```

## Conclusion
A custom agent loop lets you enforce deterministic guardrails, inject extra state fields, and control exactly when tools and models are allowed to run.
