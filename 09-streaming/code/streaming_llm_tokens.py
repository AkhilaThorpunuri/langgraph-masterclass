"""Streaming LLM Tokens -- async token streaming with astream_events.

Objective:
    Show how to stream tokens as the model generates them (before a node even
    finishes) and filter events by `metadata["langgraph_node"]`.
"""
import asyncio
import socket
from typing import TypedDict

import mlflow
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph


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


# STEP 1 -- STATE.
class TokenState(TypedDict):
    prompt: str
    response: str


fake_llm = GenericFakeChatModel(
    messages=iter([AIMessage(content="LangGraph streams tokens smoothly node by node.")])
)


# STEP 2 -- ASYNC NODE: streams tokens from the chat model and collects the final text.
async def generate_answer(state: TokenState) -> dict:
    # state coming IN:  {"prompt": "Explain streaming", "response": ""}
    # state going OUT:  {"response": "LangGraph streams tokens smoothly node by node."}
    chunks = []
    async for chunk in fake_llm.astream([HumanMessage(content=state["prompt"])]):
        chunks.append(chunk.content)
    return {"response": "".join(chunks)}


# STEP 3 -- GRAPH, EDGES & COMPILE.
builder = StateGraph(TokenState)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()


# STEP 4 -- CONSUME TOKEN EVENTS IN REAL TIME:
async def main():
    print("\n--- Streaming LLM Tokens Demo ---")
    print("  Live output: ", end="", flush=True)
    async for event in graph.astream_events({"prompt": "Explain streaming", "response": ""}, version="v2"):
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
