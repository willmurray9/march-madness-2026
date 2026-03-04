from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mm2026.utils.config import load_all_configs
from mm2026.utils.io import ensure_dir, read_csv, write_csv, write_json


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _csv_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "cols": 0,
        "columns": [],
        "null_cells": 0,
        "null_fraction": 0.0,
        "last_modified_utc": None,
    }
    if not path.exists():
        return summary

    stat = path.stat()
    summary["last_modified_utc"] = _iso_utc(stat.st_mtime)
    df = read_csv(path)
    if df.empty and len(df.columns) == 0:
        return summary

    rows = int(len(df))
    cols = int(len(df.columns))
    null_cells = int(df.isna().sum().sum())
    total_cells = rows * cols
    summary["rows"] = rows
    summary["cols"] = cols
    summary["columns"] = df.columns.tolist()
    summary["null_cells"] = null_cells
    summary["null_fraction"] = float(null_cells / total_cells) if total_cells > 0 else 0.0
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _latest_manifest(submissions_dir: Path) -> dict[str, Any]:
    manifests = sorted(submissions_dir.glob("*_manifest.json"))
    if not manifests:
        return {}
    latest = manifests[-1]
    payload = _read_json(latest)
    payload["manifest_path"] = str(latest)
    return payload


def _collect_file_summaries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = cfg["data"]
    genders = data_cfg.get("genders", ["M", "W"])
    curated_dir = Path(data_cfg["curated_dir"])
    features_dir = Path(data_cfg["features_dir"])

    rows: list[dict[str, Any]] = []

    base_files = [
        ("sample_submission_configured", curated_dir / "SampleSubmissionConfigured.csv"),
    ]
    for name, path in base_files:
        s = _csv_summary(path)
        s["category"] = "data"
        s["name"] = name
        s["gender"] = "all"
        rows.append(s)

    for gender in genders:
        data_files = [
            (f"{gender.lower()}_regular_season_long", curated_dir / f"{gender}_regular_season_long.csv"),
            (f"{gender.lower()}_tourney_compact", curated_dir / f"{gender}_tourney_compact.csv"),
            (f"{gender.lower()}_tourney_seeds", curated_dir / f"{gender}_tourney_seeds.csv"),
            (f"{gender.lower()}_seasons", curated_dir / f"{gender}_seasons.csv"),
        ]
        for name, path in data_files:
            s = _csv_summary(path)
            s["category"] = "data"
            s["name"] = name
            s["gender"] = gender
            rows.append(s)

        feat_files = [
            (f"{gender.lower()}_train_features", features_dir / f"{gender}_train_features.csv"),
            (f"{gender.lower()}_inference_features", features_dir / f"{gender}_inference_features.csv"),
            (f"{gender.lower()}_team_snapshot", features_dir / f"{gender}_team_snapshot.csv"),
        ]
        for name, path in feat_files:
            s = _csv_summary(path)
            s["category"] = "features"
            s["name"] = name
            s["gender"] = gender
            rows.append(s)
    return rows


def _config_hashes() -> dict[str, dict[str, str | None]]:
    paths = {
        "data": Path("configs/data.yaml"),
        "features": Path("configs/features.yaml"),
        "models": Path("configs/models.yaml"),
        "train": Path("configs/train.yaml"),
    }
    out: dict[str, dict[str, str | None]] = {}
    for key, path in paths.items():
        out[key] = {"path": str(path), "sha256": _sha256_file(path)}
    return out


def _train_reports(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data_cfg = cfg["data"]
    report_dir = Path(data_cfg["artifacts_dir"]) / "reports"
    out: dict[str, dict[str, Any]] = {}
    for gender in data_cfg.get("genders", ["M", "W"]):
        out[gender] = _read_json(report_dir / f"{gender}_train_report.json")
    return out


def _upsert_runs_index(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        df = read_csv(path)
    else:
        df = pd.DataFrame()

    if not df.empty and "run_id" in df.columns:
        df = df[df["run_id"] != row["run_id"]]

    merged = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    merged = merged.sort_values("generated_at_utc").reset_index(drop=True)
    write_csv(merged, path)


def run() -> None:
    cfg = load_all_configs()
    data_cfg = cfg["data"]
    artifacts_dir = ensure_dir(data_cfg["artifacts_dir"])
    reports_dir = ensure_dir(Path(artifacts_dir) / "reports")
    submissions_dir = ensure_dir(Path(artifacts_dir) / "submissions")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_reports = _train_reports(cfg)
    file_summaries = _collect_file_summaries(cfg)
    latest_submission = _latest_manifest(submissions_dir)

    payload = {
        "run_id": stamp,
        "generated_at_utc": stamp,
        "config_hashes": _config_hashes(),
        "artifacts_dir": str(artifacts_dir),
        "file_summaries": file_summaries,
        "train_reports": train_reports,
        "submission": latest_submission,
    }

    write_json(payload, reports_dir / f"observability_snapshot_{stamp}.json")
    write_json(payload, reports_dir / "observability_latest.json")

    men = train_reports.get("M", {})
    women = train_reports.get("W", {})
    m_brier = men.get("oof_brier_champion_calibrated")
    w_brier = women.get("oof_brier_champion_calibrated")
    index_row = {
        "run_id": stamp,
        "generated_at_utc": stamp,
        "m_oof_brier": m_brier,
        "w_oof_brier": w_brier,
        "mean_oof_brier": (
            (float(m_brier) + float(w_brier)) / 2.0
            if m_brier is not None and w_brier is not None
            else None
        ),
        "m_champion": men.get("champion_raw_model"),
        "w_champion": women.get("champion_raw_model"),
        "m_calibration": men.get("calibration"),
        "w_calibration": women.get("calibration"),
        "submission_rows": latest_submission.get("rows"),
        "submission_path": latest_submission.get("path"),
    }
    _upsert_runs_index(reports_dir / "runs_index.csv", index_row)

    print(f"Wrote observability snapshot: {reports_dir / f'observability_snapshot_{stamp}.json'}")
    print(f"Updated latest snapshot: {reports_dir / 'observability_latest.json'}")
    print(f"Updated runs index: {reports_dir / 'runs_index.csv'}")


if __name__ == "__main__":
    run()

