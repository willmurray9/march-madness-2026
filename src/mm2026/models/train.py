from __future__ import annotations

from pathlib import Path

from mm2026.models.pipeline import save_bundle, train_gender
from mm2026.utils.config import load_all_configs
from mm2026.utils.io import ensure_dir, read_csv, write_json


def run() -> None:
    cfg = load_all_configs()
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    train_cfg = cfg["train"]
    feat_cfg = cfg.get("features", {})
    feature_families = feat_cfg.get("feature_families", {})

    features_dir = Path(data_cfg["features_dir"])
    artifacts_dir = ensure_dir(data_cfg["artifacts_dir"])
    models_dir = ensure_dir(artifacts_dir / "models")
    reports_dir = ensure_dir(artifacts_dir / "reports")

    for gender in data_cfg.get("genders", ["M", "W"]):
        train_df = read_csv(features_dir / f"{gender}_train_features.csv")
        if train_df.empty:
            print(f"Skipping {gender}: no train features found.")
            continue

        bundle, report = train_gender(train_df, model_cfg=model_cfg, train_cfg=train_cfg)
        report["feature_families"] = feature_families
        save_bundle(bundle, models_dir / f"{gender}_bundle.joblib")
        write_json(report, reports_dir / f"{gender}_train_report.json")
        print(
            f"{gender}: trained rows={report['rows']} features={report['feature_count']} "
            f"oof_brier={report['oof_brier_champion_calibrated']:.6f}"
        )


if __name__ == "__main__":
    run()
