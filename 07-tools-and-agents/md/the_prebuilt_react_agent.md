# The Prebuilt ReAct Agent
> Venkata Bhattaram (c) 2026

## Contents
* create_react_agent
* Customizing prompts and checkpointers
* When to use the prebuilt agent vs a custom one

## Overview
When you need a standard Reason-and-Act (ReAct) tool-calling agent without hand-wiring `StateGraph`, `ToolNode`, and `tools_condition`, `create_react_agent(model, tools, checkpointer=...)` builds and compiles the full loop in a single call.

Use the **prebuilt ReAct agent** for fast prototyping and standard tool-calling assistants. Move to a **custom `StateGraph`** when you need custom state fields, guardrails, multi-step approval gates, deterministic pre/post-processing nodes, or multi-agent routing.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 07-tools-and-agents/code/the_prebuilt_react_agent.py
```

```python
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
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


fake_model = ToolCapableFakeModel(
    messages=iter([
        AIMessage(content="", tool_calls=[{"id": "w1", "name": "get_weather", "args": {"city": "Hyderabad"}}]),
        AIMessage(content="It is currently 24C and sunny in Hyderabad."),
    ])
)

agent = create_react_agent(model=fake_model, tools=[get_weather], checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "react-demo-1"}}
result = agent.invoke({"messages": [HumanMessage(content="What's the weather in Hyderabad?")]}, config)
for msg in result["messages"]:
    print(f"[{type(msg).__name__}] {msg.content or msg.tool_calls}")
```

## Conclusion
`create_react_agent` gives you a production-ready tool-calling loop with checkpointer memory in a few lines, and you can swap the offline fake model for `ChatGroq`, `ChatOllama`, or `ChatOpenAI` whenever you're ready for live LLMs.
