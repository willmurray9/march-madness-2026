from __future__ import annotations

import json
from pathlib import Path

from mm2026.utils.config import load_all_configs


def run() -> None:
    cfg = load_all_configs()
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    gates = train_cfg["promotion_gates"]
    report_dir = Path(data_cfg["artifacts_dir"]) / "reports"

    for gender in data_cfg.get("genders", ["M", "W"]):
        path = report_dir / f"{gender}_train_report.json"
        if not path.exists():
            print(f"{gender}: missing report at {path}. Run `make train` first.")
            continue

        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        season_metrics = report.get("season_metrics", {})
        if not season_metrics:
            print(f"{gender}: no season metrics found in {path}")
            continue

        baseline = []
        champion = []
        regressions = []
        for season, vals in season_metrics.items():
            base_brier = float(vals.get("brier_blend_tuned", vals.get("brier_blend_equal")))
            champ_brier = float(
                vals.get(
                    "brier_champion_calibrated",
                    report.get("oof_brier_champion_calibrated", report.get("oof_brier_meta_calibrated", base_brier)),
                )
            )
            baseline.append(base_brier)
            champion.append(champ_brier)
            regressions.append(champ_brier - base_brier)
            print(
                f"{gender} season={season} baseline={base_brier:.6f} "
                f"champion={champ_brier:.6f} delta={champ_brier - base_brier:+.6f}"
            )

        mean_improvement = sum(baseline) / len(baseline) - sum(champion) / len(champion)
        worst_regression = max(regressions)
        pass_mean = mean_improvement >= float(gates["required_mean_improvement"])
        pass_worst = worst_regression <= float(gates["max_worst_season_regression"])

        status = "PASS" if pass_mean and pass_worst else "FAIL"
        print(
            f"{gender} validation={status} mean_improvement={mean_improvement:.6f} "
            f"worst_regression={worst_regression:.6f}"
        )
        if "champion_raw_model" in report:
            print(f"{gender} champion_raw_model={report['champion_raw_model']} calibration={report.get('calibration')}")


if __name__ == "__main__":
    run()
