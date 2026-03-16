from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mm2026.backtest.bracket2025 import run_bracket_backtest_2025
from mm2026.observability.explainability import run as run_explainability
from mm2026.utils.config import load_all_configs, resolve_feature_families
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


def _feature_families_by_gender(cfg: dict[str, Any]) -> dict[str, dict[str, bool]]:
    feat_cfg = cfg.get("features", {})
    genders = cfg.get("data", {}).get("genders", ["M", "W"])
    return {gender: resolve_feature_families(feat_cfg, gender) for gender in genders}


def _family_signature(families: dict[str, bool]) -> str:
    parts = [f"{k}={int(v)}" for k, v in sorted(families.items())]
    return "|".join(parts)


def _family_signature_by_gender(families_by_gender: dict[str, dict[str, bool]]) -> str:
    return "||".join(f"{gender}:{_family_signature(families)}" for gender, families in sorted(families_by_gender.items()))


def _baseline_signature() -> str:
    return _family_signature(
        {
            "advanced_rates": False,
            "sos_adjusted": False,
            "volatility": False,
            "trend": False,
            "elo_upgrades": False,
        }
    )


def _find_baseline_snapshot(
    *,
    reports_dir: Path,
    gender: str,
    oof_value: float | None,
    baseline_sig: str,
) -> tuple[str | None, dict[str, Any]]:
    runs_idx = reports_dir / "runs_index.csv"
    if not runs_idx.exists():
        return None, {}
    df = read_csv(runs_idx)
    if df.empty:
        return None, {}
    sig_col = f"{gender.lower()}_feature_family_signature"
    if sig_col not in df.columns:
        sig_col = "feature_family_signature"
    if sig_col not in df.columns:
        return None, {}
    oof_col = "m_oof_brier" if gender == "M" else "w_oof_brier"
    if oof_col not in df.columns:
        return None, {}
    cand = df[(df[sig_col] == baseline_sig) & (df[oof_col].notna())].copy()
    if cand.empty:
        return None, {}
    cand = cand.sort_values("generated_at_utc")
    row = cand.iloc[-1]
    run_id = str(row.get("run_id"))
    snap_path = row.get("snapshot_path")
    if isinstance(snap_path, str) and snap_path:
        path = Path(snap_path)
    else:
        path = reports_dir / f"observability_snapshot_{run_id}.json"
    if not path.exists():
        return run_id, {}
    payload = _read_json(path)
    return run_id, payload


def _promotion_for_gender(
    *,
    gender: str,
    train_report: dict[str, Any],
    baseline_report: dict[str, Any],
    baseline_run_id: str | None,
    gates: dict[str, Any],
) -> dict[str, Any]:
    current_oof = train_report.get("oof_brier_champion_calibrated")
    baseline_oof = baseline_report.get("oof_brier_champion_calibrated") if baseline_report else None
    if current_oof is None:
        return {"status": "unavailable"}
    if baseline_oof is None:
        return {"status": "no_baseline", "baseline_run_id": baseline_run_id}

    current_oof_f = float(current_oof)
    baseline_oof_f = float(baseline_oof)
    lift = baseline_oof_f - current_oof_f
    min_lift = float(gates.get("min_material_lift", 0.0))

    cur_season = train_report.get("season_metrics", {}) if isinstance(train_report, dict) else {}
    base_season = baseline_report.get("season_metrics", {}) if isinstance(baseline_report, dict) else {}
    shared = sorted(set(cur_season.keys()) & set(base_season.keys()), key=lambda x: int(x))
    non_worse = 0
    checked = 0
    season_rows: list[dict[str, Any]] = []
    for s in shared:
        cur_vals = cur_season.get(s, {})
        base_vals = base_season.get(s, {})
        cur_b = cur_vals.get("brier_champion_calibrated")
        base_b = base_vals.get("brier_champion_calibrated")
        if cur_b is None or base_b is None:
            continue
        cur_b_f = float(cur_b)
        base_b_f = float(base_b)
        checked += 1
        if cur_b_f <= base_b_f:
            non_worse += 1
        season_rows.append(
            {
                "season": int(s),
                "current_brier": cur_b_f,
                "baseline_brier": base_b_f,
                "delta": cur_b_f - base_b_f,
            }
        )

    min_non_worse = int(gates.get("min_non_worse_seasons", checked))
    promoted = lift >= min_lift and non_worse >= min_non_worse
    return {
        "status": "ok",
        "baseline_run_id": baseline_run_id,
        "baseline_oof_brier": baseline_oof_f,
        "current_oof_brier": current_oof_f,
        "lift": lift,
        "min_material_lift": min_lift,
        "non_worse_seasons": non_worse,
        "shared_seasons_checked": checked,
        "min_non_worse_seasons": min_non_worse,
        "promoted": bool(promoted),
        "season_deltas": season_rows,
    }


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
    families_by_gender = _feature_families_by_gender(cfg)
    family_sig = _family_signature_by_gender(families_by_gender)
    baseline_sig = _baseline_signature()
    gates = cfg.get("train", {}).get("promotion_gates", {})
    explainability_reports = run_explainability(cfg)
    file_summaries = _collect_file_summaries(cfg)
    latest_submission = _latest_manifest(submissions_dir)
    bracket_backtest_path = reports_dir / "bracket_2025_backtest.json"
    backtests: dict[str, Any] = {}

    try:
        bracket_payload = run_bracket_backtest_2025(cfg)
        write_json(bracket_payload, bracket_backtest_path)
        m_metrics = bracket_payload.get("genders", {}).get("M", {}).get("metrics", {})
        w_metrics = bracket_payload.get("genders", {}).get("W", {}).get("metrics", {})
        backtests["bracket_2025"] = {
            "path": str(bracket_backtest_path),
            "generated_at_utc": bracket_payload.get("generated_at_utc"),
            "season": bracket_payload.get("season"),
            "men_brier_overall": m_metrics.get("brier_overall"),
            "women_brier_overall": w_metrics.get("brier_overall"),
        }
    except Exception as exc:
        backtests["bracket_2025"] = {"error": str(exc)}

    payload = {
        "run_id": stamp,
        "generated_at_utc": stamp,
        "feature_families": families_by_gender,
        "feature_family_signature": family_sig,
        "feature_family_signatures_by_gender": {
            gender: _family_signature(families)
            for gender, families in families_by_gender.items()
        },
        "is_baseline_feature_set": all(
            _family_signature(families) == baseline_sig for families in families_by_gender.values()
        ),
        "config_hashes": _config_hashes(),
        "artifacts_dir": str(artifacts_dir),
        "file_summaries": file_summaries,
        "train_reports": train_reports,
        "explainability_reports": explainability_reports,
        "submission": latest_submission,
        "backtests": backtests,
    }

    promotions: dict[str, Any] = {}
    baseline_runs: dict[str, str | None] = {}
    for gender in data_cfg.get("genders", ["M", "W"]):
        current_report = train_reports.get(gender, {})
        baseline_run_id, baseline_snapshot = _find_baseline_snapshot(
            reports_dir=reports_dir,
            gender=gender,
            oof_value=current_report.get("oof_brier_champion_calibrated") if isinstance(current_report, dict) else None,
            baseline_sig=baseline_sig,
        )
        baseline_runs[gender] = baseline_run_id
        baseline_train = baseline_snapshot.get("train_reports", {}).get(gender, {}) if baseline_snapshot else {}
        promotions[gender] = _promotion_for_gender(
            gender=gender,
            train_report=current_report if isinstance(current_report, dict) else {},
            baseline_report=baseline_train if isinstance(baseline_train, dict) else {},
            baseline_run_id=baseline_run_id,
            gates=gates if isinstance(gates, dict) else {},
        )
    payload["promotion"] = {"gates": gates, "baseline_signature": baseline_sig, "by_gender": promotions}

    snapshot_path = reports_dir / f"observability_snapshot_{stamp}.json"
    write_json(payload, snapshot_path)
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
        "feature_family_signature": family_sig,
        "m_feature_family_signature": _family_signature(families_by_gender.get("M", {})),
        "w_feature_family_signature": _family_signature(families_by_gender.get("W", {})),
        "feature_families_json": json.dumps(families_by_gender, sort_keys=True),
        "is_baseline_feature_set": all(
            _family_signature(families) == baseline_sig for families in families_by_gender.values()
        ),
        "baseline_run_m": baseline_runs.get("M"),
        "baseline_run_w": baseline_runs.get("W"),
        "m_lift_vs_baseline": promotions.get("M", {}).get("lift"),
        "w_lift_vs_baseline": promotions.get("W", {}).get("lift"),
        "m_promoted": promotions.get("M", {}).get("promoted"),
        "w_promoted": promotions.get("W", {}).get("promoted"),
        "submission_rows": latest_submission.get("rows"),
        "submission_path": latest_submission.get("path"),
        "snapshot_path": str(snapshot_path),
    }
    _upsert_runs_index(reports_dir / "runs_index.csv", index_row)

    print(f"Wrote observability snapshot: {reports_dir / f'observability_snapshot_{stamp}.json'}")
    print(f"Updated latest snapshot: {reports_dir / 'observability_latest.json'}")
    print(f"Updated runs index: {reports_dir / 'runs_index.csv'}")


if __name__ == "__main__":
    run()
