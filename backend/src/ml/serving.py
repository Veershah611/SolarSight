"""Run the underperformance detector on uploaded data-logger exports.

The uploaded window is its own reference: capacity is each inverter's p99.5 output in
the window, and its usual ratio to its peers is the window median. A fault that lasts
less than half the window therefore shows up; one covering most of it is absorbed into
the baseline, so uploads should span several days (a week or more is ideal).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import polars as pl
import xgboost as xgb

from src.ml.detection import (
    KEY, DetectorConfig, add_expected, add_shortfall, prepare_frame, run_detector,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
CHART_DAYS = 7
ZERO_PU = 0.01
STATUS_RISK = {"ALARM": "HIGH", "WATCH": "MEDIUM", "OK": "LOW"}


@dataclass
class Detector:
    model: xgb.XGBRegressor
    cfg: DetectorConfig
    meta: dict


def load_detector(model_dir: Path = MODEL_DIR) -> Detector | None:
    model_path, config_path = model_dir / "expected_power.json", model_dir / "detector.json"
    if not (model_path.exists() and config_path.exists()):
        return None
    meta = json.loads(config_path.read_text())
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    cfg = DetectorConfig(k=meta["k"], h=meta["h"], min_expected_pu=meta["min_expected_pu"],
                         min_peers=meta["min_peers"])
    return Detector(model=model, cfg=cfg, meta=meta)


def _ratio(rows: pl.DataFrame) -> float | None:
    expected = rows["expected_pu"].sum()
    return round(float(rows["pu"].sum() / expected), 3) if rows.height and expected > 0 else None


def _string_imbalance(rows: pl.DataFrame) -> float | None:
    """Mean weakest/strongest string current over daylight rows, if string data exists."""
    strings = [c for c in rows.columns if c.startswith("smu_string")]
    if len(strings) < 2 or rows.is_empty():
        return None
    ratio = (pl.min_horizontal(strings) / pl.max_horizontal(strings)).alias("r")
    vals = rows.filter(pl.max_horizontal(strings) > 0).select(ratio)["r"]
    return round(float(vals.mean()), 3) if vals.len() else None


def _evidence(judged: pl.DataFrame, latest: pl.DataFrame) -> list[dict]:
    ev = [{
        "signal": "energy_vs_expected",
        "value": _ratio(latest), "typical": 1.0,
        "description": "Output on the latest day divided by what the model expected from the peers",
    }, {
        "signal": "zero_output_hours",
        "value": int(latest.filter(pl.col("pu") < ZERO_PU).height), "typical": 0,
        "description": "Daylight hours on the latest day with no output while peers were producing",
    }, {
        "signal": "output_vs_peers",
        "value": _median_ratio(latest), "typical": _median_ratio(judged),
        "description": "Median ratio of this inverter's per-unit output to its peers' (typical = whole upload)",
    }]
    if "inv_temp" in judged.columns and judged["peer_median_temp"].drop_nulls().len():
        delta = (pl.col("inv_temp") - pl.col("peer_median_temp")).mean()
        ev.append({
            "signal": "temp_vs_peers_c",
            "value": _round(latest.select(delta).item()), "typical": _round(judged.select(delta).item()),
            "description": "Inverter temperature minus the peers' median, °C",
        })
    imbalance_now, imbalance_typ = _string_imbalance(latest), _string_imbalance(judged)
    if imbalance_now is not None:
        ev.append({
            "signal": "string_balance",
            "value": imbalance_now, "typical": imbalance_typ,
            "description": "Weakest / strongest string current (1.0 = all strings equal)",
        })
    return ev


def _median_ratio(rows: pl.DataFrame) -> float | None:
    vals = rows.filter(pl.col("peer_median_pu") > 0.05).select(pl.col("pu") / pl.col("peer_median_pu"))
    return _round(vals.to_series().median()) if vals.height else None


def _round(v, nd=3):
    return None if v is None else round(float(v), nd)


def analyse(hourly: pl.DataFrame, det: Detector) -> dict:
    """Per-inverter status, evidence and chart series for an uploaded window."""
    n_inverters = hourly[KEY].n_unique()
    if n_inverters < det.cfg.min_peers + 1:
        raise ValueError(f"Need at least {det.cfg.min_peers + 1} inverters to compare against peers; "
                         f"the upload has {n_inverters}.")

    df = prepare_frame(hourly, reference_until=None)
    df = add_shortfall(add_expected(df, det.model), det.cfg)
    scored = run_detector(df, det.cfg)
    if scored.is_empty():
        raise ValueError("No daylight hours with enough peer output to judge; upload a longer or sunnier window.")
    df = df.join(scored.select(KEY, "timestamp", "cusum", "alarm"), on=[KEY, "timestamp"], how="left")

    latest_day = scored["timestamp"].max().date()
    chart_from = scored["timestamp"].max() - timedelta(days=CHART_DAYS)
    results = []
    for (inv,), rows in df.sort("timestamp").group_by(KEY, maintain_order=True):
        judged = rows.filter(pl.col("residual").is_not_null())
        latest = judged.filter(pl.col("timestamp").dt.date() == latest_day)
        alarms = judged.filter(pl.col("alarm").fill_null(False))
        ratio_latest = _ratio(latest)
        if latest["alarm"].any():
            status = "ALARM"
        elif alarms.height or (ratio_latest is not None and ratio_latest < 0.9):
            status = "WATCH"
        else:
            status = "OK"
        daily = (judged.group_by(pl.col("timestamp").dt.date().alias("day"))
                 .agg((pl.col("pu").sum() / pl.col("expected_pu").sum()).alias("energy_ratio"),
                      pl.col("alarm").any())
                 .sort("day"))
        chart = rows.filter(pl.col("timestamp") >= chart_from).select(
            "timestamp", pl.col("pu").round(4).alias("actual_pu"), pl.col("expected_pu").round(4),
            pl.col("cusum").round(3), pl.col("alarm").fill_null(False))
        results.append({
            "inverter_id": inv,
            "status": status,
            "risk_label": STATUS_RISK[status],
            "risk_score": round(max(0.0, 1 - ratio_latest) * 100, 1) if ratio_latest is not None else 0.0,
            "energy_ratio_latest_day": ratio_latest,
            "energy_ratio_window": _ratio(judged),
            "alarm_hours": alarms.height,
            "first_alarm": str(alarms["timestamp"].min()) if alarms.height else None,
            "last_alarm": str(alarms["timestamp"].max()) if alarms.height else None,
            "judged_hours": judged.height,
            "evidence": _evidence(judged, latest),
            "daily": [{"day": str(d), "energy_ratio": _round(r), "alarm": bool(a)} for d, r, a in daily.iter_rows()],
            "hourly": [{"timestamp": str(t), "actual_pu": a, "expected_pu": e, "cusum": c, "alarm": bool(al)}
                       for t, a, e, c, al in chart.iter_rows()],
        })

    order = {"ALARM": 0, "WATCH": 1, "OK": 2}
    results.sort(key=lambda r: (order[r["status"]], -r["risk_score"]))
    return {
        "summary": {
            "inverters": n_inverters,
            "period_start": str(hourly["timestamp"].min()),
            "period_end": str(hourly["timestamp"].max()),
            "latest_day": str(latest_day),
            "judged_hours": scored.height,
            "inverters_in_alarm": sum(r["status"] == "ALARM" for r in results),
            "inverters_on_watch": sum(r["status"] == "WATCH" for r in results),
            "detector": {"k": det.cfg.k, "h": det.cfg.h, "trained_until": det.meta.get("trained_until")},
        },
        "results": results,
    }
