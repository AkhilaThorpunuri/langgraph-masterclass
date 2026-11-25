# Tool Calling Basics
> Venkata Bhattaram (c) 2026

## Contents
* Defining tools with @tool
* Binding tools to a model
* ToolMessage

## Overview
Tools let a language model interact with the outside world—calling calculators, databases, APIs, or search engines. In LangChain and LangGraph:
* **`@tool`** turns a Python function (with type hints and a docstring) into a structured tool schema that models can invoke.
* **Model turn (`AIMessage`)**: instead of answering directly, the model emits `tool_calls` specifying `id`, `name`, and `args`.
* **Tool execution (`ToolMessage`)**: your graph executes the requested tool and appends a `ToolMessage(content=..., tool_call_id=...)` to `MessagesState`, then routes back to the model so it can formulate the final answer.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 07-tools-and-agents/code/tool_calling_basics.py
```

```python
import mlflow
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


TOOLS_BY_NAME = {multiply.name: multiply}


def call_model(state: MessagesState) -> dict:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"The product is {last.content}.")]}
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "multiply", "args": {"a": 6, "b": 7}}],
            )
        ]
    }


def run_tools(state: MessagesState) -> dict:
    ai_msg = state["messages"][-1]
    outputs = [
        ToolMessage(content=str(TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])), tool_call_id=tc["id"])
        for tc in ai_msg.tool_calls
    ]
    return {"messages": outputs}


def should_continue(state: MessagesState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", run_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
graph = builder.compile()

out = graph.invoke({"messages": [HumanMessage(content="What is 6 times 7?")]})
for msg in out["messages"]:
    print(f"[{type(msg).__name__}] {msg.content or msg.tool_calls}")
```

## Conclusion
Every LangGraph agent loop is built on this two-node cycle: `model` emits `tool_calls`, `tools` runs them and appends `ToolMessage`s, and a conditional edge exits to `END` once the model responds without tool calls.
