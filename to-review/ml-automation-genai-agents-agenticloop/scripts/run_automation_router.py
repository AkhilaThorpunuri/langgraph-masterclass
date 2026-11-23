"""CLI: run the automation router that moves new source-data files into the
right target-data subfolder.

Usage:
  python scripts/run_automation_router.py --once
  python scripts/run_automation_router.py --watch --poll-interval 5
"""
import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.automation_router.watcher import scan_once, watch_loop


@mlflow.trace(name="run_automation_router")
def _run_once():
    return scan_once()


def main():
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("masterclass")
    mlflow.langchain.autolog()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    parser.add_argument("--poll-interval", type=int, default=5)
    args = parser.parse_args()

    if args.once:
        results = _run_once()
        if not results:
            print("No new files to route.")
        for result in results:
            print(f"routed: {result['filename']} -> {result['decision']['target_folder']}")
    else:
        watch_loop(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
