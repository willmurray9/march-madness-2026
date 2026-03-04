from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from mm2026.models.pipeline import load_bundle, predict_gender
from mm2026.utils.ids import parse_matchup_id


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


def _dataset_description_map() -> dict[str, dict[str, str]]:
    return {
        "sample_submission_configured": {
            "contains": "Submission IDs required by the active stage file (season/team matchups).",
            "used_for": "Defines exact prediction coverage and final ID validation.",
        },
        "m_regular_season_long": {
            "contains": "Men regular-season games in team-game long format with score/outcome context.",
            "used_for": "Builds rolling team form and efficiency features.",
        },
        "w_regular_season_long": {
            "contains": "Women regular-season games in team-game long format with score/outcome context.",
            "used_for": "Builds rolling team form and efficiency features.",
        },
        "m_tourney_compact": {
            "contains": "Men NCAA tournament historical game outcomes.",
            "used_for": "Creates labeled training matchups (`target`).",
        },
        "w_tourney_compact": {
            "contains": "Women NCAA tournament historical game outcomes.",
            "used_for": "Creates labeled training matchups (`target`).",
        },
        "m_tourney_seeds": {
            "contains": "Men tournament seed assignments by season/team.",
            "used_for": "Adds seed-based matchup features.",
        },
        "w_tourney_seeds": {
            "contains": "Women tournament seed assignments by season/team.",
            "used_for": "Adds seed-based matchup features.",
        },
        "m_seasons": {
            "contains": "Men season metadata.",
            "used_for": "Season context and sanity checks.",
        },
        "w_seasons": {
            "contains": "Women season metadata.",
            "used_for": "Season context and sanity checks.",
        },
        "m_train_features": {
            "contains": "Men historical tournament matchup feature deltas plus `target`.",
            "used_for": "Trains/validates men models with rolling holdouts.",
        },
        "w_train_features": {
            "contains": "Women historical tournament matchup feature deltas plus `target`.",
            "used_for": "Trains/validates women models with rolling holdouts.",
        },
        "m_inference_features": {
            "contains": "Men Stage submission matchup feature deltas, no target.",
            "used_for": "Generates men prediction probabilities for submission.",
        },
        "w_inference_features": {
            "contains": "Women Stage submission matchup feature deltas, no target.",
            "used_for": "Generates women prediction probabilities for submission.",
        },
        "m_team_snapshot": {
            "contains": "Men latest season team-level snapshot stats at inference cutoff.",
            "used_for": "Explains feature state feeding inference matchup deltas.",
        },
        "w_team_snapshot": {
            "contains": "Women latest season team-level snapshot stats at inference cutoff.",
            "used_for": "Explains feature state feeding inference matchup deltas.",
        },
    }


def _model_method_description_map() -> dict[str, str]:
    return {
        "logistic": "Regularized linear baseline on matchup feature deltas.",
        "hgb": "Histogram gradient-boosted trees for non-linear tabular patterns.",
        "rf": "Random forest ensemble over tabular features.",
        "xgb": "XGBoost boosted trees.",
        "catboost": "CatBoost boosted trees.",
        "elo": "Hand-crafted Elo probability curve from Elo rating deltas.",
        "blend": "Weighted average of base model probabilities.",
        "meta": "Stacked logistic meta-model trained on base model outputs.",
        "none": "No probability calibration.",
        "platt": "Logistic (Platt) calibration of raw probabilities.",
        "isotonic": "Isotonic calibration of raw probabilities.",
    }


def _dataset_overview_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    files_df = pd.DataFrame(snapshot.get("file_summaries", []))
    if files_df.empty:
        return pd.DataFrame()
    desc_map = _dataset_description_map()
    files_df["contains"] = files_df["name"].map(lambda x: desc_map.get(str(x), {}).get("contains", ""))
    files_df["used_for"] = files_df["name"].map(lambda x: desc_map.get(str(x), {}).get("used_for", ""))
    return files_df.sort_values(["category", "gender", "name"]).reset_index(drop=True)


def _model_overview_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    desc = _model_method_description_map()
    for gender, report in snapshot.get("train_reports", {}).items():
        if not isinstance(report, dict) or not report:
            continue
        enabled_models = report.get("enabled_models", [])
        for m in enabled_models:
            rows.append(
                {
                    "gender": gender,
                    "method": m,
                    "role": "base_model",
                    "selected": True,
                    "description": desc.get(m, ""),
                }
            )
        rows.append(
            {
                "gender": gender,
                "method": report.get("champion_raw_model"),
                "role": "final_raw_selector",
                "selected": True,
                "description": desc.get(str(report.get("champion_raw_model")), ""),
            }
        )
        rows.append(
            {
                "gender": gender,
                "method": report.get("calibration"),
                "role": "calibration",
                "selected": True,
                "description": desc.get(str(report.get("calibration")), ""),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates()
    return out


def _load_submission_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    sub_path = snapshot.get("submission", {}).get("path")
    if not sub_path:
        return pd.DataFrame()
    p = Path(sub_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "Pred" not in df.columns:
        return pd.DataFrame()
    return df


def _load_bracket_backtest(snapshot: dict[str, Any]) -> dict[str, Any]:
    backtest_path = snapshot.get("backtests", {}).get("bracket_2025", {}).get("path")
    if not backtest_path:
        return {}
    path = Path(backtest_path)
    if not path.exists():
        return {}
    return _load_json(path)


def _load_explainability_payload(snapshot: dict[str, Any], gender: str) -> dict[str, Any]:
    reports = snapshot.get("explainability_reports", {})
    report = reports.get(gender, {}) if isinstance(reports, dict) else {}
    path = report.get("path") if isinstance(report, dict) else None
    if path and Path(path).exists():
        return _load_json(Path(path))
    return report if isinstance(report, dict) else {}


def _round_sort_key(label: str) -> int:
    order = {
        "Play-In": 0,
        "Round of 64": 1,
        "Round of 32": 2,
        "Sweet 16": 3,
        "Elite 8": 4,
        "Final Four": 5,
        "Championship": 6,
    }
    return order.get(str(label), 99)


def _bracket_table_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df
    keep = [
        "round_label",
        "slot",
        "team_low_name",
        "team_low_seed",
        "team_high_name",
        "team_high_seed",
        "pred_team_low_win",
        "winner_team_name",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.sort_values(["round_label", "slot"], key=lambda s: s.map(_round_sort_key) if s.name == "round_label" else s)
    rename = {
        "round_label": "Round",
        "slot": "Slot",
        "team_low_name": "Team Low",
        "team_low_seed": "Low Seed",
        "team_high_name": "Team High",
        "team_high_seed": "High Seed",
        "pred_team_low_win": "P(Low Wins)",
        "winner_team_name": "Winner",
    }
    return out.rename(columns=rename)


def _brier_example_df() -> pd.DataFrame:
    demo = pd.DataFrame(
        {
            "y_true": [1, 0, 1, 0, 1],
            "y_pred": [0.84, 0.30, 0.62, 0.20, 0.53],
        }
    )
    demo["squared_error"] = (demo["y_pred"] - demo["y_true"]) ** 2
    return demo


@st.cache_data
def _load_team_name_map() -> dict[int, str]:
    out: dict[int, str] = {}
    for path in [Path("data/raw/latest/MTeams.csv"), Path("data/raw/latest/WTeams.csv")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "TeamID" not in df.columns or "TeamName" not in df.columns:
            continue
        for row in df.itertuples(index=False):
            out[int(row.TeamID)] = str(row.TeamName)
    return out


@st.cache_data
def _load_seed_map() -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for path in [Path("data/curated/latest/M_tourney_seeds.csv"), Path("data/curated/latest/W_tourney_seeds.csv")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Season" not in df.columns or "TeamID" not in df.columns:
            continue
        seed_col = "seed_num" if "seed_num" in df.columns else "Seed"
        if seed_col not in df.columns:
            continue
        if seed_col == "Seed":
            parsed = df["Seed"].astype(str).str.extract(r"(\d{2})", expand=False)
            df = df.assign(seed_num=pd.to_numeric(parsed, errors="coerce"))
            seed_col = "seed_num"
        for row in df[["Season", "TeamID", seed_col]].dropna().itertuples(index=False):
            out[(int(row.Season), int(row.TeamID))] = float(getattr(row, seed_col))
    return out


def _infer_gender(low_id: int, high_id: int) -> str:
    if 1000 <= low_id < 2000 and 1000 <= high_id < 2000:
        return "M"
    if 3000 <= low_id < 4000 and 3000 <= high_id < 4000:
        return "W"
    return "?"


def _enrich_submission_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ID" not in df.columns:
        return df
    team_names = _load_team_name_map()
    seeds = _load_seed_map()
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        mid = parse_matchup_id(str(row.ID))
        team_low = int(mid.team_low)
        team_high = int(mid.team_high)
        season = int(mid.season)
        rows.append(
            {
                "ID": row.ID,
                "Season": season,
                "Gender": _infer_gender(team_low, team_high),
                "TeamLowID": team_low,
                "TeamLowName": team_names.get(team_low, f"Team {team_low}"),
                "TeamLowSeed": seeds.get((season, team_low)),
                "TeamHighID": team_high,
                "TeamHighName": team_names.get(team_high, f"Team {team_high}"),
                "TeamHighSeed": seeds.get((season, team_high)),
                "Pred": float(row.Pred),
            }
        )
    return pd.DataFrame(rows)


def _seeded_matchups_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["TeamLowSeed"].notna() & df["TeamHighSeed"].notna()].copy()


@st.cache_data
def _brier_with_real_matchups(max_rows_per_gender: int = 6000) -> tuple[pd.DataFrame, str | None]:
    team_names = _load_team_name_map()
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    for gender in ["M", "W"]:
        feat_path = Path(f"data/features/latest/{gender}_train_features.csv")
        bundle_path = Path(f"artifacts/models/{gender}_bundle.joblib")
        if not feat_path.exists() or not bundle_path.exists():
            continue

        df = pd.read_csv(feat_path)
        if df.empty or "target" not in df.columns:
            continue

        # Keep this bounded for dashboard responsiveness.
        if len(df) > max_rows_per_gender:
            df = df.sample(n=max_rows_per_gender, random_state=42).sort_index()

        bundle = load_bundle(bundle_path)

        model_cfg = {"base_models": {"elo": {"scale": 400.0}}}
        preds = predict_gender(bundle, df, model_cfg=model_cfg)
        scored = df[["Season", "TeamID_low", "TeamID_high", "target", "low_seed_num", "high_seed_num"]].copy()
        scored["pred"] = preds
        scored["squared_error"] = (scored["pred"] - scored["target"].astype(float)) ** 2
        scored["Gender"] = gender
        scored["TeamLowName"] = scored["TeamID_low"].map(lambda t: team_names.get(int(t), f"Team {int(t)}"))
        scored["TeamHighName"] = scored["TeamID_high"].map(lambda t: team_names.get(int(t), f"Team {int(t)}"))
        out.append(scored)

    warning_text = "\n".join(warnings) if warnings else None
    if not out:
        return pd.DataFrame(), warning_text
    return pd.concat(out, ignore_index=True), warning_text


def main() -> None:
    st.set_page_config(page_title="MM2026 Observability", layout="wide")
    st.title("MM2026 Modeling Overview")
    st.caption("Datasets, modeling methods, prediction behavior, and Brier score calculation.")

    default_reports = Path("artifacts") / "reports"
    reports_dir = Path(st.sidebar.text_input("Reports directory", str(default_reports)))
    st.sidebar.markdown("Run `make observe` after data/features/train/submit to refresh.")

    snapshot = _load_latest_snapshot(reports_dir)
    if not snapshot:
        st.error(f"No observability snapshot found in {reports_dir}.")
        st.stop()

    dataset_df = _dataset_overview_df(snapshot)
    methods_df = _model_overview_df(snapshot)
    submission_df = _load_submission_df(snapshot)
    submission_enriched = _enrich_submission_rows(submission_df)
    submission_seeded = _seeded_matchups_only(submission_enriched)
    pred_hist = _prediction_summary(snapshot)
    season_df = _season_rows(snapshot)
    train_df = _train_rows(snapshot)
    bracket_backtest = _load_bracket_backtest(snapshot)
    explainability = {g: _load_explainability_payload(snapshot, g) for g in ["M", "W"]}

    top = st.columns(4)
    top[0].metric("Snapshot UTC", snapshot.get("generated_at_utc", "unknown"))
    top[1].metric("Datasets tracked", int(len(dataset_df)))
    top[2].metric("Submission rows", int(len(submission_enriched)))
    mean_brier = float(train_df["oof_brier_champion_calibrated"].mean()) if not train_df.empty else float("nan")
    top[3].metric("Mean OOF Brier (M/W)", f"{mean_brier:.6f}" if not np.isnan(mean_brier) else "n/a")

    st.subheader("1) Dataset Overview")
    if dataset_df.empty:
        st.info("No dataset summaries found. Run `make observe`.")
    else:
        cols = ["name", "gender", "category", "rows", "cols", "contains", "used_for", "path"]
        st.dataframe(dataset_df[cols], width="stretch")

    st.subheader("2) Modeling Methods")
    if methods_df.empty:
        st.info("No train reports found.")
    else:
        st.dataframe(methods_df[["gender", "role", "method", "description"]], width="stretch")
        st.caption("Selected per-gender model details")
        st.dataframe(
            train_df[
                [
                    "gender",
                    "rows",
                    "feature_count",
                    "champion_raw_model",
                    "calibration",
                    "oof_brier_champion_calibrated",
                ]
            ],
            width="stretch",
        )

    st.subheader("3) Explainability")
    for gender in ["M", "W"]:
        report = explainability.get(gender, {})
        st.markdown(f"**{gender} Explainability**")
        if not report or report.get("status") != "ok":
            st.info(f"No explainability report for {gender}. Run `make observe` after `make train`.")
            continue

        top = report.get("top_features", {})
        perm_top = pd.DataFrame(top.get("permutation", []))
        coef_top = pd.DataFrame(top.get("logistic_abs_coef", []))
        shap_top = top.get("shap_mean_abs", {})
        artifacts = report.get("artifacts", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout Rows", str(report.get("rows_scored", "n/a")))
        c2.metric("Feature Count", str(report.get("feature_count", "n/a")))
        c3.metric("Champion Raw Model", str(report.get("champion_raw_model", "n/a")))

        if not perm_top.empty:
            st.caption("Permutation importance (mean Brier increase when permuted)")
            st.dataframe(perm_top, width="stretch")
        if not coef_top.empty:
            st.caption("Logistic coefficient importance (absolute coefficient)")
            st.dataframe(coef_top, width="stretch")

        shap_artifacts = artifacts.get("shap", {}) if isinstance(artifacts, dict) else {}
        for model_name in ["hgb", "xgb", "catboost"]:
            model_payload = shap_artifacts.get(model_name, {})
            if not model_payload:
                continue
            st.caption(f"SHAP ({model_name})")
            if isinstance(shap_top, dict):
                top_df = pd.DataFrame(shap_top.get(model_name, []))
                if not top_df.empty:
                    st.dataframe(top_df, width="stretch")
            for key in ["bar_plot", "beeswarm_plot"]:
                path = model_payload.get(key)
                if path and Path(path).exists():
                    st.image(str(path), caption=f"{model_name} {key.replace('_', ' ')}", use_container_width=True)

    st.subheader("4) Prediction Visualization")
    if submission_seeded.empty:
        st.info("No submission file found in latest snapshot.")
    else:
        st.caption(
            f"Showing only seeded-vs-seeded rows (tournament-like matchups): "
            f"{len(submission_seeded):,} of {len(submission_enriched):,} total rows."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        q = submission_seeded["Pred"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        c1.metric("Min Pred", f"{submission_seeded['Pred'].min():.4f}")
        c2.metric("Median Pred", f"{q[0.5]:.4f}")
        c3.metric("Max Pred", f"{submission_seeded['Pred'].max():.4f}")
        c4.metric("P05", f"{q[0.05]:.4f}")
        c5.metric("P95", f"{q[0.95]:.4f}")

        seeded_hist = pd.cut(
            submission_seeded["Pred"],
            bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            include_lowest=True,
        ).value_counts(sort=False)
        seeded_hist_df = seeded_hist.reset_index()
        seeded_hist_df.columns = ["bin", "count"]
        seeded_hist_df["bin"] = seeded_hist_df["bin"].astype(str)
        st.bar_chart(seeded_hist_df.set_index("bin"))

        st.caption(
            "To avoid repeated teams from simple sorting, examples below are random samples from confidence bands."
        )
        ui1, ui2, ui3 = st.columns(3)
        random_seed = int(ui1.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1))
        sample_n = int(ui2.slider("Rows per sample", min_value=5, max_value=50, value=15, step=1))
        confident_pct = float(ui3.slider("Confident band (top % by Pred)", min_value=0.5, max_value=10.0, value=2.0, step=0.5))

        st.caption("Confident picks sample")
        conf_cut = float(submission_seeded["Pred"].quantile(1 - confident_pct / 100.0))
        confident_pool = submission_seeded[submission_seeded["Pred"] >= conf_cut].copy()
        if confident_pool.empty:
            confident_show = submission_seeded.sort_values("Pred", ascending=False).head(sample_n)
        else:
            confident_show = confident_pool.sample(n=min(sample_n, len(confident_pool)), random_state=random_seed)
        confident_show = confident_show.sort_values("Pred", ascending=False)
        st.dataframe(
            confident_show[
                ["Season", "Gender", "TeamLowName", "TeamLowSeed", "TeamHighName", "TeamHighSeed", "Pred"]
            ],
            width="stretch",
        )

        st.caption("Uncertain picks sample (closest to 0.5)")
        uncertain = submission_seeded.assign(uncertainty=(submission_seeded["Pred"] - 0.5).abs())
        uncertain_pool_cut = float(uncertain["uncertainty"].quantile(0.1))
        uncertain_pool = uncertain[uncertain["uncertainty"] <= uncertain_pool_cut].copy()
        if uncertain_pool.empty:
            uncertain_show = uncertain.sort_values("uncertainty", ascending=True).head(sample_n)
        else:
            uncertain_show = uncertain_pool.sample(n=min(sample_n, len(uncertain_pool)), random_state=random_seed + 1)
        uncertain_show = uncertain_show.sort_values("uncertainty", ascending=True)
        st.dataframe(
            uncertain_show[
                ["Season", "Gender", "TeamLowName", "TeamLowSeed", "TeamHighName", "TeamHighSeed", "Pred"]
            ],
            width="stretch",
        )

    st.subheader("5) How Brier Score Is Calculated")
    st.code("Brier = mean((y_pred - y_true)^2)", language="text")
    st.caption(
        "Lower is better. Perfect predictions have Brier = 0.0. "
        "In this repo, `target=1` means TeamLow (lower TeamID in matchup ID) won; `target=0` means TeamHigh won. "
        "`pred` is model probability that TeamLow wins."
    )

    demo = _brier_example_df()
    st.write("Worked example (toy numbers)")
    st.dataframe(demo, width="stretch")
    st.metric("Toy Example Brier", f"{demo['squared_error'].mean():.6f}")

    st.write("Worked example (real historical matchups + team names/seeds)")
    brier_real, brier_warning = _brier_with_real_matchups()
    if brier_warning:
        st.warning(brier_warning)
    if brier_real.empty:
        st.info("Real matchup Brier sample unavailable. Run `make train` to generate model bundles and train features.")
    else:
        sample_n = min(20, len(brier_real))
        real_sample = brier_real.sort_values("squared_error", ascending=False).head(sample_n)
        st.dataframe(
            real_sample[
                [
                    "Gender",
                    "Season",
                    "TeamLowName",
                    "low_seed_num",
                    "TeamHighName",
                    "high_seed_num",
                    "target",
                    "pred",
                    "squared_error",
                ]
            ].rename(columns={"low_seed_num": "TeamLowSeed", "high_seed_num": "TeamHighSeed"}),
            width="stretch",
        )
        st.metric("Real Sample Mean Squared Error", f"{real_sample['squared_error'].mean():.6f}")

    st.write("Your actual run metrics (from training reports)")
    if train_df.empty:
        st.info("No training metrics found.")
    else:
        st.dataframe(
            train_df[
                [
                    "gender",
                    "oof_brier_blend_raw",
                    "oof_brier_meta_raw",
                    "oof_brier_champion_raw",
                    "oof_brier_champion_calibrated",
                ]
            ],
            width="stretch",
        )
    if not season_df.empty and "brier_champion_calibrated" in season_df.columns:
        piv = season_df.pivot(index="season", columns="gender", values="brier_champion_calibrated").sort_index()
        st.caption("Per-season champion calibrated Brier")
        st.line_chart(piv)
        st.dataframe(
            season_df[["gender", "season", "brier_blend_equal", "brier_blend_tuned", "brier_champion_calibrated"]]
            .sort_values(["season", "gender"]),
            width="stretch",
        )

    st.subheader("6) 2025 Bracket Retrospective")
    if not bracket_backtest:
        st.info("No 2025 bracket backtest artifact found. Run `make observe`.")
    else:
        season = bracket_backtest.get("season", 2025)
        st.caption(
            f"Leakage-safe setup: train on seasons <= 2024; infer season {season} bracket with cutoff DayNum "
            f"{bracket_backtest.get('daynum_cutoff', 'n/a')}."
        )
        for gender in ["M", "W"]:
            gender_payload = bracket_backtest.get("genders", {}).get(gender, {})
            if not gender_payload:
                st.warning(f"No backtest payload for {gender}.")
                continue
            metrics = gender_payload.get("metrics", {})
            predicted_games = gender_payload.get("predicted_games", [])
            actual_games = gender_payload.get("actual_games", [])

            st.markdown(f"**{gender} Tournament**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            brier_overall = metrics.get("brier_overall")
            mc1.metric("Brier (overall)", f"{brier_overall:.6f}" if isinstance(brier_overall, (int, float)) else "n/a")
            champ_hit = metrics.get("champion_hit")
            mc2.metric("Champion Hit", "n/a" if champ_hit is None else ("Yes" if champ_hit else "No"))
            mc3.metric("Final Four Overlap", str(metrics.get("final_four_overlap_count", "n/a")))
            mc4.metric("Games Scored", str(metrics.get("games_total", "n/a")))

            st.caption("Predicted bracket")
            pred_df = _bracket_table_df(predicted_games)
            if pred_df.empty:
                st.info(f"No predicted bracket rows for {gender}.")
            else:
                st.dataframe(pred_df, width="stretch")

            st.caption("Actual bracket")
            act_df = _bracket_table_df(actual_games)
            if act_df.empty:
                st.info(f"No actual bracket rows for {gender}.")
            else:
                st.dataframe(act_df, width="stretch")

            brier_rows = []
            for round_label, round_brier in metrics.get("brier_by_round", {}).items():
                brier_rows.append(
                    {
                        "Round": round_label,
                        "Games": int(metrics.get("games_by_round", {}).get(round_label, 0)),
                        "Brier": float(round_brier),
                    }
                )
            if brier_rows:
                brier_df = pd.DataFrame(brier_rows).sort_values("Round", key=lambda s: s.map(_round_sort_key))
                st.caption("Per-round Brier (realized 2025 bracket games)")
                st.dataframe(brier_df, width="stretch")


if __name__ == "__main__":
    main()
