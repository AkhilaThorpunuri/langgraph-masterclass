"""The Prebuilt ReAct Agent -- using create_react_agent with tools and memory.

Demonstrates:
- Building a ReAct agent with langgraph.prebuilt.create_react_agent
- Using a lightweight deterministic Runnable chat model so the script runs offline
  (swap `model` for ChatGroq / ChatOllama / ChatOpenAI in production)
"""
from typing import Any

import mlflow
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"{city}: 24C, sunny"


class ToolCapableFakeModel(GenericFakeChatModel):
    """Minimal offline chat model that supports bind_tools() for create_react_agent."""

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


fake_model = ToolCapableFakeModel(
    messages=iter([
        AIMessage(
            content="",
            tool_calls=[{"id": "w1", "name": "get_weather", "args": {"city": "Hyderabad"}}],
        ),
        AIMessage(content="It is currently 24C and sunny in Hyderabad."),
    ])
)

agent = create_react_agent(
    model=fake_model,
    tools=[get_weather],
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "react-demo-1"}}
result = agent.invoke({"messages": [HumanMessage(content="What's the weather in Hyderabad?")]}, config)
for msg in result["messages"]:
    print(f"[{type(msg).__name__}] {msg.content or msg.tool_calls}")
