from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mm2026.models.pipeline import load_bundle, predict_gender
from mm2026.utils.config import load_all_configs
from mm2026.utils.io import ensure_dir, read_csv, write_csv, write_json


def _validate_submission(df: pd.DataFrame) -> None:
    if set(df.columns) != {"ID", "Pred"}:
        raise ValueError("Submission must contain exactly ID and Pred columns.")
    if df["ID"].isna().any() or df["Pred"].isna().any():
        raise ValueError("Submission contains null values.")
    if df["ID"].duplicated().any():
        raise ValueError("Submission contains duplicate IDs.")
    if ((df["Pred"] < 0) | (df["Pred"] > 1)).any():
        raise ValueError("Pred values must be in [0, 1].")


def _validate_submission_against_sample(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if sample.empty:
        return
    if sample["ID"].duplicated().any():
        raise ValueError("Sample submission contains duplicate IDs.")
    if len(submission) != len(sample):
        raise ValueError(f"Submission row count mismatch: got={len(submission)}, expected={len(sample)}")
    expected_ids = set(sample["ID"].tolist())
    got_ids = set(submission["ID"].tolist())
    if expected_ids != got_ids:
        missing = len(expected_ids - got_ids)
        extra = len(got_ids - expected_ids)
        raise ValueError(f"Submission ID mismatch: missing={missing}, extra={extra}")


def run() -> None:
    cfg = load_all_configs()
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]

    features_dir = Path(data_cfg["features_dir"])
    artifacts_dir = ensure_dir(data_cfg["artifacts_dir"])
    model_dir = Path(artifacts_dir) / "models"
    submissions_dir = ensure_dir(Path(artifacts_dir) / "submissions")

    all_parts = []
    for gender in data_cfg.get("genders", ["M", "W"]):
        inf_df = read_csv(features_dir / f"{gender}_inference_features.csv")
        if inf_df.empty:
            print(f"Skipping {gender}: no inference features found.")
            continue

        bundle = load_bundle(model_dir / f"{gender}_bundle.joblib")
        preds = predict_gender(bundle, inf_df, model_cfg=model_cfg)
        part = pd.DataFrame({"ID": inf_df["ID"], "Pred": preds})
        all_parts.append(part)

    if not all_parts:
        raise ValueError("No inference predictions generated.")

    submission = pd.concat(all_parts, ignore_index=True)

    sample_cfg_path = Path(data_cfg["sample_submission_file"])
    sample = read_csv(sample_cfg_path)
    _validate_submission_against_sample(submission, sample)

    submission = submission.sort_values("ID").reset_index(drop=True)
    _validate_submission(submission)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_label = "stage2" if "stage2" in sample_cfg_path.name.lower() else "stage1"
    out_path = submissions_dir / f"{stamp}_{stage_label}.csv"
    write_csv(submission, out_path)

    manifest = {
        "generated_at_utc": stamp,
        "rows": int(len(submission)),
        "path": str(out_path),
    }
    write_json(manifest, submissions_dir / f"{stamp}_manifest.json")
    print(f"Wrote submission: {out_path}")


if __name__ == "__main__":
    run()
