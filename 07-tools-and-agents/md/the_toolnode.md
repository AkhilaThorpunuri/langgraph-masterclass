# The ToolNode
> Venkata Bhattaram (c) 2026

## Contents
* Prebuilt ToolNode
* Routing with tools_condition
* Error handling in tools

## Overview
Writing the tool-dispatch loop and routing function by hand every time is repetitive. LangGraph provides two drop-in building blocks in `langgraph.prebuilt`:
* **`ToolNode(tools, handle_tool_errors=True)`**: automatically reads `tool_calls` from the latest `AIMessage`, invokes each tool (in parallel when multiple calls are present), wraps results into `ToolMessage`s, and catches exceptions cleanly instead of crashing the graph.
* **`tools_condition`**: a prebuilt routing function that returns `"tools"` when the latest message has `tool_calls` and `END` otherwise.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 07-tools-and-agents/code/the_toolnode.py
```

```python
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


def agent(state: MessagesState) -> dict:
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


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([divide], handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

print(graph.invoke({"messages": [HumanMessage(content="Divide 20 by 4")]})["messages"][-1].content)
print(graph.invoke({"messages": [HumanMessage(content="Divide 20 by zero")]})["messages"][-1].content)
```

## Conclusion
`ToolNode` and `tools_condition` eliminate boilerplate and make tool failures recoverable—the error text is passed back to the agent as a `ToolMessage` so the model can self-correct or inform the user.
