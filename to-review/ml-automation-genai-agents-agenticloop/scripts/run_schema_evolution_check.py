"""CLI: check config/schema_registry.yaml for newly activated tables/columns
(evolution) or newly deactivated ones (rollback). The graph's load_handler
node then routes + rejoins target-data itself whenever it acts, so a single
run of this script is a complete, self-contained pass of the agentic loop.
"""
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema_evolution.graph import run


@mlflow.trace(name="run_schema_evolution_check")
def _run_schema_check():
    return run()


def main():
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("masterclass")
    mlflow.langchain.autolog()

    result = _run_schema_check()
    if result["regenerated"]:
        print("Schema evolution detected -- regenerated data:", result["changes"])
    elif result["archived"]:
        print("Rollback detected -- no active columns/tables added:", result["changes"])
    else:
        print("No schema changes detected.")

    if result["archived"]:
        print("Archived rolled-back tables:")
        for table_name, path in result["archived"].items():
            print(f" - {table_name}: {path}")

    if result["routed_files"] or result["load_outputs"]:
        print(f"load_handler: routed {result['routed_files']} file(s); joined output:")
        for kind, path in result["load_outputs"].items():
            print(f" - {kind}: {path}")


if __name__ == "__main__":
    main()
