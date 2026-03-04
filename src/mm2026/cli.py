from __future__ import annotations

import argparse

from mm2026.data.ingest import run as run_data
from mm2026.evaluation.validate import run as run_validate
from mm2026.features.build import run as run_features
from mm2026.models.train import run as run_train
from mm2026.observability.explainability import run as run_explainability
from mm2026.observability.report import run as run_observe
from mm2026.submission.generate import run as run_submit
from mm2026.utils.config import load_all_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="March Madness 2026 pipeline")
    parser.add_argument("command", choices=["data", "features", "train", "validate", "submit", "explain", "observe"])
    args = parser.parse_args()

    if args.command == "data":
        run_data()
    elif args.command == "features":
        run_features()
    elif args.command == "train":
        run_train()
    elif args.command == "validate":
        run_validate()
    elif args.command == "submit":
        run_submit()
    elif args.command == "explain":
        run_explainability(load_all_configs())
    elif args.command == "observe":
        run_observe()


if __name__ == "__main__":
    main()
