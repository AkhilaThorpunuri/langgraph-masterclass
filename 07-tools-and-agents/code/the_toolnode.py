"""The ToolNode -- prebuilt ToolNode, tools_condition, and tool error handling.

Objective:
    Replace manual tool dispatch and routing with LangGraph's prebuilt ToolNode
    and tools_condition, including graceful error handling on ValueError.
"""
import socket

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
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


# STEP 1 -- TOOL: raises ValueError on division by zero to demonstrate error handling.
@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


tools = [divide]


# STEP 2 -- AGENT NODE.
def agent(state: MessagesState) -> dict:
    # Turn 1 IN:  [HumanMessage(...)]    -> OUT: [AIMessage(tool_calls=[divide(20, 4 or 0)])]
    # Turn 2 IN:  [..., ToolMessage(...)] -> OUT: [AIMessage("Tool replied: ...")]
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"Tool replied: {last.content}")]}
    divisor = 0 if "zero" in last.content.lower() else 4
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "div_1", "name": "divide", "args": {"a": 20, "b": divisor}}],
            )
        ]
    }


# STEP 3 -- GRAPH & PREBUILT TOOLNODE:
# ToolNode(handle_tool_errors=True) catches exceptions and returns them as ToolMessages.
# tools_condition routes to "tools" if tool_calls exist, otherwise END.
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

# STEP 4 -- INVOKE BOTH NORMAL AND ERROR CASES:
print("\n--- ToolNode Demo ---")
res1 = graph.invoke({"messages": [HumanMessage(content="Divide 20 by 4")]})
print(f"  Normal (20 / 4) -> {res1['messages'][-1].content}")

res2 = graph.invoke({"messages": [HumanMessage(content="Divide 20 by zero")]})
print(f"  Error  (20 / 0) -> {res2['messages'][-1].content}")
