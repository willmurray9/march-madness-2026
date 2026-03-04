from __future__ import annotations

import argparse

from mm2026.data.ingest import run as run_data
from mm2026.evaluation.validate import run as run_validate
from mm2026.features.build import run as run_features
from mm2026.models.train import run as run_train
from mm2026.submission.generate import run as run_submit


def main() -> None:
    parser = argparse.ArgumentParser(description="March Madness 2026 pipeline")
    parser.add_argument("command", choices=["data", "features", "train", "validate", "submit"])
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


if __name__ == "__main__":
    main()
