"""Tool Calling Basics -- @tool, AIMessage tool_calls, and ToolMessage.

Objective:
    Show the fundamental two-node agent cycle: model emits tool_calls ->
    tools node executes them and appends a ToolMessage -> model replies.
"""
import socket

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph


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


# STEP 1 -- TOOL: @tool turns a typed Python function into a callable tool schema.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


TOOLS_BY_NAME = {multiply.name: multiply}


# STEP 2 -- NODES (using built-in MessagesState, which appends messages via add_messages).
def call_model(state: MessagesState) -> dict:
    # Turn 1 IN:  [HumanMessage("What is 6 times 7?")] -> OUT: [AIMessage(tool_calls=[multiply(6, 7)])]
    # Turn 2 IN:  [..., ToolMessage("42")]             -> OUT: [AIMessage("The product is 42.")]
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
    # state coming IN:  [..., AIMessage(tool_calls=[...])]
    # state going OUT: [ToolMessage(content="42", tool_call_id="call_1")]
    ai_msg = state["messages"][-1]
    outputs = []
    for tc in ai_msg.tool_calls:
        result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
        outputs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": outputs}


# STEP 3 -- ROUTING FUNCTION: loop to "tools" if model requested tool_calls, else END.
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# STEP 4 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", run_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
graph = builder.compile()

# STEP 5 -- INVOKE: watch the full Human -> AI(tool_call) -> ToolMessage -> AI(final) cycle.
print("\n--- Tool Calling Basics Demo ---")
out = graph.invoke({"messages": [HumanMessage(content="What is 6 times 7?")]})
for msg in out["messages"]:
    role = type(msg).__name__.replace("Message", "")
    if getattr(msg, "tool_calls", None):
        tc = msg.tool_calls[0]
        print(f"  {role:<6} -> calls {tc['name']}({tc['args']})")
    else:
        print(f"  {role:<6} -> {msg.content}")
