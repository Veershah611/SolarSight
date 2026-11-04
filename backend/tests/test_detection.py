from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.ml.detection import (
    DetectorConfig, add_expected, add_peer_context, add_per_unit, add_shortfall,
    cusum, fit_expected_power, prepare_frame, run_detector,
)

START = datetime(2024, 3, 1)


def plant(days=30, sizes=(100.0, 100.0, 50.0, 80.0), fault=None, seed=0):
    """Inverters of different sizes following a shared clear-sky curve with cloud noise.

    fault: (inverter index, first day, scale) multiplies that inverter's output from that day on.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        for h in range(24):
            sun = max(0.0, np.sin(np.pi * (h - 6) / 12)) * rng.uniform(0.7, 1.0)
            for i, size in enumerate(sizes):
                power = size * sun * rng.normal(1.0, 0.02)
                if fault and i == fault[0] and d >= fault[1]:
                    power *= fault[2]
                rows.append({"inverter_id": f"INV{i}", "plant_id": "p", "hour": h,
                             "timestamp": START + timedelta(days=d, hours=h),
                             "inv_power": max(power, 0.0), "inv_temp": 30 + 20 * sun})
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))


def test_peer_context_excludes_the_inverter_itself():
    df = add_peer_context(add_per_unit(plant(days=2, fault=(0, 0, 0.0)), {f"INV{i}": s for i, s in enumerate((100, 100, 50, 80))}))
    noon = df.filter((pl.col("hour") == 12) & (pl.col("inverter_id") == "INV0"))
    # INV0 is dead, but its peers are producing, so its peer median stays high
    assert noon["pu"].max() == 0
    assert noon["peer_median_pu"].min() > 0.5
    assert noon["peer_count"].to_list() == [3, 3]


def test_cusum_alarms_on_sustained_shortfall_not_on_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, 500)
    assert not cusum(noise, k=0.1, h=2.0)[1].any()
    shortfall = np.concatenate([noise[:100], noise[100:200] - 0.3])
    alarms = np.flatnonzero(cusum(shortfall, k=0.1, h=2.0)[1])
    assert alarms.size and 100 <= alarms[0] <= 115


def test_detector_flags_degraded_inverter_only():
    df = prepare_frame(plant(days=40, fault=(2, 30, 0.6)), reference_until=START + timedelta(days=25))
    train = df.filter(pl.col("timestamp") < START + timedelta(days=25))
    model = fit_expected_power(train.with_columns(pl.lit(False).alias("is_outage_hour")))
    cfg = DetectorConfig(k=0.1, h=2.0)
    scored = run_detector(add_shortfall(add_expected(df, model), cfg), cfg)
    alarms = scored.filter(pl.col("alarm"))
    assert set(alarms["inverter_id"]) == {"INV2"}
    assert alarms["timestamp"].min() >= START + timedelta(days=30)


def test_reference_ratio_absorbs_size_and_baseline_matches_peers():
    df = prepare_frame(plant(days=10), reference_until=None)
    judged = add_shortfall(add_expected(df, None), DetectorConfig())
    residual = judged["residual"].drop_nulls()
    assert abs(residual.median()) < 0.05
