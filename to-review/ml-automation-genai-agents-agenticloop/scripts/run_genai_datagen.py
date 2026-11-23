"""CLI: run the genAI data-generation LangGraph program.

Usage: python scripts/run_genai_datagen.py [--count N]
"""
import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.genai_datagen.graph import run


@mlflow.trace(name="run_genai_datagen")
def _run_datagen(count: int):
    return run(count=count)


def main():
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("masterclass")
    mlflow.langchain.autolog()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60, help="Number of products to generate")
    args = parser.parse_args()

    result = _run_datagen(count=args.count)
    print("Generated files:")
    for f in result["written_files"]:
        print(f" - {f}")


if __name__ == "__main__":
    main()
