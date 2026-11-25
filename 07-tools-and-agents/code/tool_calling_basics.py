"""Tool Calling Basics -- @tool, AIMessage tool_calls, and ToolMessage.

Demonstrates:
- Defining tools with @tool
- Simulating a tool-calling model turn that emits AIMessage(tool_calls=...)
- Executing the tool and returning a ToolMessage back into MessagesState
"""
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
    # Emit a structured tool call (what a tool-bound chat model produces)
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
    outputs = []
    for tc in ai_msg.tool_calls:
        result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
        outputs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": outputs}


def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


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
