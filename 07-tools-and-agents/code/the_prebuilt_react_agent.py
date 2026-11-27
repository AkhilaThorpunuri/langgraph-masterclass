"""The Prebuilt ReAct Agent -- using create_react_agent with tools and memory.

Objective:
    Show how create_react_agent builds and compiles a complete ReAct tool loop
    with checkpointer memory in one call (using an offline fake model here).
"""
import socket
from typing import Any

import mlflow
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


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


# STEP 1 -- TOOL.
@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"{city}: 24C, sunny"


# STEP 2 -- MODEL: deterministic offline stand-in (swap for ChatGroq/ChatOllama/ChatOpenAI).
class ToolCapableFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


fake_model = ToolCapableFakeModel(
    messages=iter([
        AIMessage(content="", tool_calls=[{"id": "w1", "name": "get_weather", "args": {"city": "Hyderabad"}}]),
        AIMessage(content="It is currently 24C and sunny in Hyderabad."),
    ])
)

# STEP 3 -- BUILD PREBUILT AGENT: wires agent <-> tools loop + MemorySaver automatically.
agent = create_react_agent(model=fake_model, tools=[get_weather], checkpointer=MemorySaver())

# STEP 4 -- INVOKE:
print("\n--- Prebuilt ReAct Agent Demo ---")
config = {"configurable": {"thread_id": "react-demo-1"}}
result = agent.invoke({"messages": [HumanMessage(content="What's the weather in Hyderabad?")]}, config)
for msg in result["messages"]:
    role = type(msg).__name__.replace("Message", "")
    if getattr(msg, "tool_calls", None):
        tc = msg.tool_calls[0]
        print(f"  {role:<6} -> calls {tc['name']}({tc['args']})")
    else:
        print(f"  {role:<6} -> {msg.content}")
