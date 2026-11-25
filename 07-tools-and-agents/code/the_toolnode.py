"""The ToolNode -- prebuilt ToolNode, tools_condition, and tool error handling.

Demonstrates:
- Replacing a manual tool loop with langgraph.prebuilt.ToolNode
- Routing with langgraph.prebuilt.tools_condition
- Graceful error handling when a tool raises ValueError
"""
import mlflow
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


tools = [divide]


def agent(state: MessagesState) -> dict:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"Tool replied: {last.content}")]}
    text = last.content.lower()
    divisor = 0 if "zero" in text else 4
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "div_1", "name": "divide", "args": {"a": 20, "b": divisor}}],
            )
        ]
    }


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

print("Normal division:")
res1 = graph.invoke({"messages": [HumanMessage(content="Divide 20 by 4")]})
print(" ", res1["messages"][-1].content)

print("Division by zero (handled by ToolNode):")
res2 = graph.invoke({"messages": [HumanMessage(content="Divide 20 by zero")]})
print(" ", res2["messages"][-1].content)
