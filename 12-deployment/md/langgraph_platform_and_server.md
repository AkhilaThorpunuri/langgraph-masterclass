# LangGraph Platform and Server
> Venkata Bhattaram (c) 2026

## Contents
* langgraph.json config
* Local dev server
* Exporting a compiled graph

## Overview
To serve a LangGraph application with `langgraph dev` or deploy it to **LangGraph Platform**, you expose a module-level compiled `graph` variable and register it in a `langgraph.json` manifest:
```json
{
  "dependencies": ["."],
  "graphs": {
    "echo": "./langgraph_platform_and_server.py:graph"
  },
  "env": ".env"
}
```
The server automatically wraps your compiled graph with HTTP endpoints for threads, runs, streaming, and checkpoints.

## Code Example
Run the standalone script (with MLflow tracing to `http://localhost:5000/` under experiment `masterclass`):
```powershell
python 12-deployment/code/langgraph_platform_and_server.py
```

```python
import json
from typing import TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("masterclass")
mlflow.langchain.autolog()


class EchoState(TypedDict):
    message: str
    reply: str


def respond(state: EchoState) -> dict:
    return {"reply": f"LangGraph Server received: {state['message']}"}


builder = StateGraph(EchoState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

graph = builder.compile()

if __name__ == "__main__":
    print(graph.invoke({"message": "Hello Platform!", "reply": ""}))
```

## Conclusion
Keep your graph construction clean and export the compiled `graph` at module level so `langgraph.json` can load it for local dev, Studio, or cloud deployment without extra glue code.
