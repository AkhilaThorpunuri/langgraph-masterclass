"""Streaming LLM Tokens -- async token streaming with astream_events and custom writers.

Demonstrates:
- Streaming tokens/chunks as they are generated using stream_mode="custom" and astream_events()
- Filtering streamed events by node name
"""
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
    chunks = []
    async for chunk in fake_llm.astream([HumanMessage(content=state["prompt"])]):
        chunks.append(chunk.content)
    return {"response": "".join(chunks)}


builder = StateGraph(TokenState)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()


async def main():
    print("Streaming tokens from node 'generate_answer': ", end="", flush=True)
    async for event in graph.astream_events(
        {"prompt": "Explain streaming", "response": ""},
        version="v2",
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == "generate_answer"
        ):
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
