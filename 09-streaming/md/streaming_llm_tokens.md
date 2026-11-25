# Streaming LLM Tokens
> Venkata Bhattaram (c) 2026

## Contents
* astream_events
* Token-by-token output
* Filtering events by node

## Overview
Node-level streaming (`stream_mode="updates"`) still waits for each node's LLM call to finish before yielding. To stream tokens *as the model generates them*, use **`graph.astream_events(..., version="v2")`** and filter for `on_chat_model_stream` events.

Every event includes `metadata["langgraph_node"]`, so in multi-node graphs you can stream tokens only from your user-facing answer node while hiding internal router or classifier tokens.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 09-streaming/code/streaming_llm_tokens.py
```

```python
import asyncio
from typing import TypedDict

import mlflow
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class TokenState(TypedDict):
    prompt: str
    response: str


fake_llm = GenericFakeChatModel(
    messages=iter([AIMessage(content="LangGraph streams tokens smoothly node by node.")])
)


async def generate_answer(state: TokenState) -> dict:
    chunks = [chunk.content async for chunk in fake_llm.astream([HumanMessage(content=state["prompt"])])]
    return {"response": "".join(chunks)}


builder = StateGraph(TokenState)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()


async def main():
    async for event in graph.astream_events({"prompt": "Explain streaming", "response": ""}, version="v2"):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == "generate_answer"
        ):
            print(event["data"]["chunk"].content, end="", flush=True)
    print()


asyncio.run(main())
```

## Conclusion
`astream_events(version="v2")` plus `metadata["langgraph_node"]` filtering gives you responsive, ChatGPT-style token streaming while keeping internal reasoning nodes silent.
