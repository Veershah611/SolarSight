"""Benchmark the underperformance detector (see detection.py).

Real partial faults (a dead string, soiling, derating) aren't labelled in the data,
so the detector is scored the standard way for anomaly detection: inject faults of
known size into held-out data and measure how many are caught, how fast, and how
often the detector alarms on days with no known problem. Real outage episodes are
scored too.

Both the XGBoost expected-power model and a no-ML baseline (peer median x the
inverter's usual ratio to its peers) get their own alarm threshold tuned to the same
false-alarm budget on the validation period, so the comparison is like for like.

Usage (from backend/):
    python -m src.ml.evaluate_detection --raw-dir <folder with *.raws.csv>
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.ml.dataset import load_dataset
from src.ml.detection import (
    KEY, MODEL_FEATURES, DetectorConfig, add_expected, add_shortfall, core_mask, cusum, daily_summary,
    fit_expected_power, prepare_frame, run_detector,
)
from src.ml.labels import LabelConfig, find_episodes, mark_outage_hours

BACKEND_DIR = Path(__file__).resolve().parents[2]
FAULT_SIZES = [0.1, 0.2, 0.3, 0.5, 1.0]
# CUSUM slack candidates; picked on validation by detection of small (10-20%) injected faults
K_GRID = [0.03, 0.05, 0.1, 0.15]
FAULT_DAYS = 3
LOOKBACK_DAYS = 7


# --------------------------------------------------------------------------- helpers

def regression_metrics(df: pl.DataFrame) -> dict:
    d = df.filter(core_mask() & (pl.col("expected_pu") >= 0.1) & ~pl.col("is_outage_hour")
                  & pl.col("pu").is_not_null() & pl.col("expected_pu").is_not_null())
    y, p = d["pu"].to_numpy(), d["expected_pu"].to_numpy()
    return {
        "rows": len(y),
        "mae_pu": round(float(np.mean(np.abs(y - p))), 4),
        "median_abs_pct_error": round(float(np.median(np.abs(y - p) / np.maximum(p, 0.1))), 4),
        "r2": round(float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), 4),
    }


def known_event_days(episodes: pl.DataFrame) -> set[tuple[str, object]]:
    """Inverter-days within a day of a known outage episode."""
    days = set()
    for inv, start, end in episodes.select(KEY, "start", "end").iter_rows():
        d = start.date() - timedelta(days=1)
        while d <= end.date() + timedelta(days=1):
            days.add((inv, d))
            d += timedelta(days=1)
    return days


def without_event_days(days: pl.DataFrame, event_days: set) -> pl.DataFrame:
    events = pl.DataFrame([{KEY: inv, "day": d} for inv, d in event_days],
                          schema={KEY: pl.String, "day": pl.Date})
    return days.join(events, on=[KEY, "day"], how="anti")


def false_alarm_rate(scored: pl.DataFrame, event_days: set) -> float:
    """Alarm-days per inverter-month, ignoring days around known outages."""
    clean = without_event_days(daily_summary(scored), event_days)
    return float(clean["alarm"].sum()) / max(clean.height / 30, 1e-9)


def tune_h(judged: pl.DataFrame, cfg: DetectorConfig, event_days: set, target: float) -> float:
    """Smallest CUSUM threshold (to 0.1) whose validation false-alarm rate is within budget.

    Binary search: a higher threshold never needs more evidence to stay quiet, so the
    rate falls (almost) monotonically with h.
    """
    grid = np.arange(0.1, 40.05, 0.1)
    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        rate = false_alarm_rate(run_detector(judged, DetectorConfig(k=cfg.k, h=float(grid[mid]))), event_days)
        if rate <= target:
            hi = mid
        else:
            lo = mid + 1
    return float(grid[lo])


def pick_injections(judged: pl.DataFrame, event_days: set, n: int, seed: int) -> list[tuple[str, object]]:
    """Random (inverter, onset hour) pairs with a full lookback and fault window, away from known events."""
    rng = np.random.default_rng(seed)
    t0 = judged["timestamp"].min() + timedelta(days=LOOKBACK_DAYS)
    t1 = judged["timestamp"].max() - timedelta(days=FAULT_DAYS)
    candidates = judged.filter(pl.col("timestamp").is_between(t0, t1)).select(KEY, "timestamp")
    picks = []
    while len(picks) < n:
        inv, ts = candidates.row(int(rng.integers(candidates.height)))
        window = {(inv, (ts + timedelta(days=d)).date()) for d in range(-1, FAULT_DAYS + 1)}
        if not window & event_days:
            picks.append((inv, ts))
    return picks


def injection_benchmark(judged: pl.DataFrame, injections, cfg: DetectorConfig, sizes=FAULT_SIZES) -> dict:
    by_inv = {inv: part.sort("timestamp") for (inv,), part in judged.group_by(KEY)}
    out = {}
    for size in sizes:
        caught, ttd = 0, []
        for inv, onset in injections:
            part = by_inv[inv].filter(pl.col("timestamp").is_between(
                onset - timedelta(days=LOOKBACK_DAYS), onset + timedelta(days=FAULT_DAYS), closed="left"))
            ts = part["timestamp"].to_numpy()
            pu, exp = part["pu"].to_numpy(), part["expected_pu"].to_numpy()
            in_fault = ts >= np.datetime64(onset)
            faulty = np.where(in_fault, pu * (1 - size), pu)
            residual = np.clip((faulty - exp) / exp, -1.5, 1.5)
            _, alarm = cusum(residual, cfg.k, cfg.h)
            hits = np.flatnonzero(alarm & in_fault)
            if hits.size:
                caught += 1
                ttd.append(int(hits[0] - np.argmax(in_fault)) + 1)
        out[f"{int(size * 100)}%"] = {
            "detection_rate": round(caught / len(injections), 3),
            "median_hours_to_detect": float(np.median(ttd)) if ttd else None,
        }
    return out


def real_episode_detection(scored: pl.DataFrame, episodes: pl.DataFrame, test_start, min_hours: int) -> dict:
    eps = episodes.filter((pl.col("start") >= test_start + timedelta(days=LOOKBACK_DAYS))
                          & (pl.col("outage_hours") >= min_hours))
    caught, ttd = 0, []
    for inv, start, end in eps.select(KEY, "start", "end").iter_rows():
        window = scored.filter((pl.col(KEY) == inv) & pl.col("timestamp").is_between(start, end + timedelta(days=1)))
        hit = window.filter(pl.col("alarm"))
        if hit.height:
            caught += 1
            ttd.append(window.filter(pl.col("timestamp") <= hit["timestamp"][0]).height)
    return {"episodes": eps.height, "caught": caught,
            "median_hours_to_detect": float(np.median(ttd)) if ttd else None}


# --------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=BACKEND_DIR / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "reports")
    parser.add_argument("--false-alarm-budget", type=float, default=1.0,
                        help="alarm-days per inverter-month allowed on validation")
    parser.add_argument("--injections", type=int, default=300)
    parser.add_argument("--export-model", action="store_true",
                        help="save the model and tuned CUSUM settings to models/ for the API")
    args = parser.parse_args(argv)

    raw = load_dataset(args.raw_dir, args.cache_dir)
    label_cfg = LabelConfig(min_hours=1)
    df = mark_outage_hours(raw, label_cfg)
    episodes = find_episodes(df, label_cfg)
    event_days = known_event_days(episodes)

    c1, c2 = df["timestamp"].quantile(0.6), df["timestamp"].quantile(0.8)
    df = prepare_frame(df, reference_until=c1)
    train = df.filter(pl.col("timestamp") < c1)
    val = df.filter((pl.col("timestamp") >= c1) & (pl.col("timestamp") < c2))
    test = df.filter(pl.col("timestamp") >= c2)
    print(f"train < {c1}, val < {c2}, test >= {c2}")

    model = fit_expected_power(train, val)
    report = {"periods": {"train_until": str(c1), "test_from": str(c2), "test_until": str(df["timestamp"].max())},
              "regression": {}, "cross_plant": {}, "detectors": {}}

    # Expected-power accuracy on healthy test hours
    for name, m in [("xgboost", model), ("peer_baseline", None)]:
        report["regression"][name] = regression_metrics(add_expected(test, m))
        print(name, report["regression"][name])

    # Does the expected-power model transfer to a plant it never saw?
    plants = sorted(df["plant_id"].unique().to_list())
    for held_out in plants if len(plants) > 1 else []:
        m = fit_expected_power(train.filter(pl.col("plant_id") != held_out))
        report["cross_plant"][held_out] = regression_metrics(add_expected(test.filter(pl.col("plant_id") == held_out), m))
        print("cross-plant", held_out, report["cross_plant"][held_out])

    for name, m in [("xgboost", model), ("peer_baseline", None)]:
        val_j = add_shortfall(add_expected(val, m), DetectorConfig())
        val_judged = val_j.filter(pl.col("residual").is_not_null())
        val_injections = pick_injections(val_judged, event_days, args.injections, seed=3)
        tuning = []
        for k in K_GRID:
            h = tune_h(val_j, DetectorConfig(k=k), event_days, args.false_alarm_budget)
            small = injection_benchmark(val_judged, val_injections, DetectorConfig(k=k, h=h), sizes=[0.1, 0.2])
            tuning.append((np.mean([v["detection_rate"] for v in small.values()]), k, h))
            print(f"  {name} k={k} h={h:.1f} val small-fault detection {tuning[-1][0]:.3f}")
        _, k, h = max(tuning)
        cfg = DetectorConfig(k=k, h=h)
        test_j = add_shortfall(add_expected(test, m), cfg)
        scored = run_detector(test_j, cfg)
        judged = test_j.filter(pl.col("residual").is_not_null())
        injections = pick_injections(judged, event_days, args.injections, seed=7)
        res = {
            "k": cfg.k, "h": round(h, 2),
            "test_false_alarm_days_per_inverter_month": round(false_alarm_rate(scored, event_days), 3),
            "injected_faults": injection_benchmark(judged, injections, cfg),
            "real_outages_ge_1h": real_episode_detection(scored, episodes, c2, 1),
            "real_outages_ge_3h": real_episode_detection(scored, episodes, c2, 3),
        }
        if name == "xgboost":
            days = daily_summary(scored).filter(pl.col("alarm"))
            unexplained = without_event_days(days, event_days)
            res["top_unexplained_alarm_days"] = [
                [inv, str(day), round(float(ratio), 3)]
                for inv, day, ratio in unexplained.sort("energy_ratio").select(KEY, "day", "energy_ratio").head(8).iter_rows()
            ]
        report["detectors"][name] = res
        print(name, json.dumps(res, default=str))

    if args.export_model:
        export_detector(model, report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "detection.json").write_text(json.dumps(report, indent=2, default=str))
    (args.out_dir / "detection.md").write_text(render_markdown(report, args), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'detection.md'}")


def export_detector(model, report: dict) -> None:
    """Save the benchmarked model and its tuned CUSUM settings for the API (see serving.py)."""
    model_dir = BACKEND_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model.save_model(model_dir / "expected_power.json")
    det = report["detectors"]["xgboost"]
    config = {
        "k": det["k"], "h": det["h"],
        "min_expected_pu": DetectorConfig.min_expected_pu, "min_peers": DetectorConfig.min_peers,
        "features": MODEL_FEATURES,
        "trained_until": report["periods"]["train_until"],
        "benchmark": {
            "expected_power": report["regression"]["xgboost"],
            "injected_faults": det["injected_faults"],
            "false_alarm_days_per_inverter_month": det["test_false_alarm_days_per_inverter_month"],
        },
    }
    (model_dir / "detector.json").write_text(json.dumps(config, indent=2))
    print(f"Exported detector to {model_dir}")


def render_markdown(r: dict, args) -> str:
    na = lambda v: "n/a" if v is None else v
    p = r["periods"]
    lines = [
        "# Underperformance Detection Benchmark",
        "",
        "Generated by `python -m src.ml.evaluate_detection` (see `backend/src/ml/evaluate_detection.py`).",
        "",
        f"Expected-power model trained before {p['train_until'][:10]}; detector thresholds tuned on the next 20% of "
        f"time (slack k chosen for best small-fault detection, threshold h for at most {args.false_alarm_budget:g} "
        f"alarm-day per inverter-month); everything below is scored on "
        f"{p['test_from'][:10]} to {p['test_until'][:10]}.",
        "",
        "## Expected power (healthy core-daylight hours)",
        "",
        "| Model | MAE (per-unit) | Median abs. error | R² |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for name, m in r["regression"].items():
        lines.append(f"| {name} | {m['mae_pu']} | {m['median_abs_pct_error']:.1%} | {m['r2']} |")
    for plant, m in r["cross_plant"].items():
        lines.append(f"| xgboost, never saw {plant} | {m['mae_pu']} | {m['median_abs_pct_error']:.1%} | {m['r2']} |")

    lines += [
        "",
        "## Injected faults",
        "",
        f"{args.injections} random inverter/onset pairs (away from known outages), each fault lasting "
        f"{FAULT_DAYS} days; output scaled down by the fault size. Time to detect counts core-daylight hours.",
        "",
        "| Fault size | " + " | ".join(f"{n}: detected" for n in r["detectors"]) + " | "
        + " | ".join(f"{n}: median hours" for n in r["detectors"]) + " |",
        "| :--- |" + " ---: |" * (2 * len(r["detectors"])),
    ]
    for size in next(iter(r["detectors"].values()))["injected_faults"]:
        det = [r["detectors"][n]["injected_faults"][size] for n in r["detectors"]]
        lines.append(f"| {size} | " + " | ".join(f"{d['detection_rate']:.0%}" for d in det) + " | "
                     + " | ".join(str(na(d["median_hours_to_detect"])) for d in det) + " |")

    lines += [
        "",
        "## Real outages and false alarms (test period)",
        "",
        "| Detector | CUSUM k / h | Outages ≥ 1 h caught | Outages ≥ 3 h caught | Median hours to detect (≥ 3 h) | Alarm-days / inverter-month (no known event) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, d in r["detectors"].items():
        o1, o3 = d["real_outages_ge_1h"], d["real_outages_ge_3h"]
        lines.append(f"| {name} | {d['k']} / {d['h']} | {o1['caught']}/{o1['episodes']} | {o3['caught']}/{o3['episodes']} | "
                     f"{na(o3['median_hours_to_detect'])} | {d['test_false_alarm_days_per_inverter_month']} |")

    top = r["detectors"].get("xgboost", {}).get("top_unexplained_alarm_days")
    if top:
        lines += ["", "## Largest alarms not explained by a known outage", "",
                  "Energy ratio = actual / expected output over the day's judged hours.", "",
                  "| Inverter | Day | Energy ratio |", "| :--- | :--- | ---: |"]
        lines += [f"| {inv} | {day} | {ratio} |" for inv, day, ratio in top]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
