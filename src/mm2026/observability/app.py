from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _load_latest_snapshot(reports_dir: Path) -> dict[str, Any]:
    latest = reports_dir / "observability_latest.json"
    if latest.exists():
        return _load_json(latest)
    candidates = sorted(reports_dir.glob("observability_snapshot_*.json"))
    if not candidates:
        return {}
    return _load_json(candidates[-1])


def _train_rows(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gender, report in snapshot.get("train_reports", {}).items():
        if not report:
            continue
        rows.append(
            {
                "gender": gender,
                "rows": report.get("rows"),
                "feature_count": report.get("feature_count"),
                "oof_brier_champion_calibrated": report.get("oof_brier_champion_calibrated"),
                "oof_brier_champion_raw": report.get("oof_brier_champion_raw"),
                "oof_brier_blend_raw": report.get("oof_brier_blend_raw"),
                "oof_brier_meta_raw": report.get("oof_brier_meta_raw"),
                "champion_raw_model": report.get("champion_raw_model"),
                "calibration": report.get("calibration"),
            }
        )
    return pd.DataFrame(rows)


def _season_rows(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gender, report in snapshot.get("train_reports", {}).items():
        season_metrics = report.get("season_metrics", {}) if isinstance(report, dict) else {}
        for season, metrics in season_metrics.items():
            row = {"gender": gender, "season": int(season)}
            if isinstance(metrics, dict):
                row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_summary(snapshot: dict[str, Any]) -> pd.DataFrame:
    submission = snapshot.get("submission", {})
    path = submission.get("path")
    if not path:
        return pd.DataFrame()
    sub_path = Path(path)
    if not sub_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(sub_path)
    if df.empty or "Pred" not in df.columns:
        return pd.DataFrame()

    bins = pd.cut(
        df["Pred"],
        bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )
    counts = bins.value_counts(sort=False)
    out = counts.reset_index()
    out.columns = ["bin", "count"]
    out["bin"] = out["bin"].astype(str)
    return out


def main() -> None:
    st.set_page_config(page_title="MM2026 Observability", layout="wide")
    st.title("MM2026 Pipeline Observability")
    st.caption("End-to-end view of data, features, models, and submissions.")

    default_reports = Path("artifacts") / "reports"
    reports_dir = Path(st.sidebar.text_input("Reports directory", str(default_reports)))
    st.sidebar.markdown("Run `make observe` after data/features/train/submit to refresh.")

    snapshot = _load_latest_snapshot(reports_dir)
    if not snapshot:
        st.error(f"No observability snapshot found in {reports_dir}.")
        st.stop()

    runs_index_path = reports_dir / "runs_index.csv"
    runs_index = pd.read_csv(runs_index_path) if runs_index_path.exists() else pd.DataFrame()
    files_df = pd.DataFrame(snapshot.get("file_summaries", []))
    train_df = _train_rows(snapshot)
    season_df = _season_rows(snapshot)
    pred_hist = _prediction_summary(snapshot)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshot UTC", snapshot.get("generated_at_utc", "unknown"))
    c2.metric("Runs tracked", int(len(runs_index)))
    c3.metric("Files summarized", int(len(files_df)))
    c4.metric("Submission rows", snapshot.get("submission", {}).get("rows", 0))

    st.subheader("Run History")
    if runs_index.empty:
        st.info("No runs index yet. Run `make observe` to create one.")
    else:
        st.dataframe(runs_index.sort_values("generated_at_utc", ascending=False), use_container_width=True)
        plot_cols = [c for c in ["m_oof_brier", "w_oof_brier", "mean_oof_brier"] if c in runs_index.columns]
        if plot_cols:
            hist_df = runs_index[["generated_at_utc"] + plot_cols].set_index("generated_at_utc")
            st.line_chart(hist_df)

    st.subheader("Pipeline Assets")
    if files_df.empty:
        st.info("No file summaries found in snapshot.")
    else:
        st.dataframe(
            files_df[["category", "gender", "name", "exists", "rows", "cols", "null_fraction", "last_modified_utc", "path"]],
            use_container_width=True,
        )

    st.subheader("Training Overview")
    if train_df.empty:
        st.info("Training reports not found in snapshot.")
    else:
        st.dataframe(train_df, use_container_width=True)

    st.subheader("Season Metrics")
    if season_df.empty:
        st.info("No season-level metrics available.")
    else:
        available = [c for c in ["brier_blend_equal", "brier_blend_tuned", "brier_champion_calibrated"] if c in season_df.columns]
        metric = st.selectbox("Metric", options=available if available else season_df.columns.tolist())
        piv = season_df.pivot(index="season", columns="gender", values=metric).sort_index()
        st.line_chart(piv)
        st.dataframe(season_df.sort_values(["season", "gender"]), use_container_width=True)

    st.subheader("Model Details")
    for gender in ["M", "W"]:
        report = snapshot.get("train_reports", {}).get(gender, {})
        if not report:
            continue
        with st.expander(f"{gender} model details", expanded=False):
            st.write(f"Champion: `{report.get('champion_raw_model')}`")
            st.write(f"Calibration: `{report.get('calibration')}`")

            blend_weights = report.get("blend_weights", {})
            if isinstance(blend_weights, dict) and blend_weights:
                bw = pd.DataFrame([{"model": k, "weight": v} for k, v in blend_weights.items()]).set_index("model")
                st.bar_chart(bw)

            cal_scores = report.get("calibration_candidates", {})
            if isinstance(cal_scores, dict) and cal_scores:
                st.dataframe(
                    pd.DataFrame([{"method": k, "brier": v} for k, v in cal_scores.items()]).sort_values("brier"),
                    use_container_width=True,
                )

    st.subheader("Prediction Distribution")
    if pred_hist.empty:
        st.info("No submission prediction file found in snapshot.")
    else:
        st.bar_chart(pred_hist.set_index("bin"))
        st.dataframe(pred_hist, use_container_width=True)


if __name__ == "__main__":
    main()

