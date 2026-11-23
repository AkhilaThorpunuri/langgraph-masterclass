"""CLI: run the data-loader LangGraph program that joins all target-data
tables into one denormalized dataset.
"""
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.graph import run


@mlflow.trace(name="run_data_loader")
def _run_loader():
    return run()


def main():
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("masterclass")
    mlflow.langchain.autolog()

    result = _run_loader()
    print("Joined output written to:")
    for kind, path in result["outputs"].items():
        print(f" - {kind}: {path}")


if __name__ == "__main__":
    main()
