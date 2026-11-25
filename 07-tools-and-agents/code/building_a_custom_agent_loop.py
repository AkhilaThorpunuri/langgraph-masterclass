"""Building a Custom Agent Loop -- agent node + tool node + memory + guardrails.

Demonstrates:
- Custom agent loop with an explicit guardrail node before the agent runs
- Checkpointed memory across turns on the same thread_id
- Tool execution via ToolNode and conditional routing
"""
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
        return {
            "blocked": True,
            "messages": [AIMessage(content="Blocked by guardrail: unsafe request detected.")],
        }
    return {"blocked": False}


def route_after_guardrail(state: GuardedAgentState) -> str:
    return END if state.get("blocked") else "agent"


def agent_node(state: GuardedAgentState) -> dict:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"Here is your update: {last.content}")]}
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "ord_1", "name": "lookup_order", "args": {"order_id": "ORD-1042"}}],
            )
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

r1 = graph.invoke({"messages": [HumanMessage(content="Check status of ORD-1042")], "blocked": False}, cfg)
print("Safe query ->", r1["messages"][-1].content)

r2 = graph.invoke({"messages": [HumanMessage(content="Please run DROP TABLE users")], "blocked": False}, cfg)
print("Unsafe query ->", r2["messages"][-1].content)
