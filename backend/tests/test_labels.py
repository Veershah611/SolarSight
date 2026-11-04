from datetime import datetime, timedelta

import polars as pl
import pytest

from src.ml.evaluate import time_split
from src.ml.labels import LabelConfig, add_targets, find_episodes, mark_outage_hours

START = datetime(2024, 3, 1)
CFG = LabelConfig(min_hours=3, merge_gap_hours=24, lead_min_days=1, lead_max_days=10)


def plant(days=40, outages=(), plant_wide=(), inverters=("A", "B", "C")):
    """Hourly rows for a 3-inverter plant producing 50 kW from 06-18h.

    outages: (inverter, day, first_hour, last_hour) with zero power for that inverter only.
    plant_wide: (day, first_hour, last_hour) with zero power for every inverter.
    """
    rows = []
    for d in range(days):
        for h in range(24):
            ts = START + timedelta(days=d, hours=h)
            for inv in inverters:
                power = 50.0 if 6 <= h <= 18 else 0.0
                if any(inv == o[0] and d == o[1] and o[2] <= h <= o[3] for o in outages):
                    power = 0.0
                if any(d == p[0] and p[1] <= h <= p[2] for p in plant_wide):
                    power = 0.0
                rows.append({"inverter_id": inv, "plant_id": "p", "timestamp": ts, "hour": h, "inv_power": power})
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))


def labelled(df, cfg=CFG):
    df = mark_outage_hours(df, cfg)
    episodes = find_episodes(df, cfg)
    return add_targets(df, episodes, cfg), episodes


def test_night_and_dawn_zero_power_is_not_an_outage():
    df = mark_outage_hours(plant(days=2), CFG)
    assert df["is_outage_hour"].sum() == 0


def test_plant_wide_outage_is_not_an_inverter_failure():
    df = mark_outage_hours(plant(days=2, plant_wide=[(1, 9, 14)]), CFG)
    assert df["is_outage_hour"].sum() == 0


def test_zero_during_plant_wide_trickle_is_not_an_outage():
    # Peers at 1% of capacity (heavy overcast/curtailment) don't count as "producing"
    df = plant(days=2, outages=[("A", 1, 8, 16)])
    trickle = (pl.col("timestamp").dt.day() == 2) & pl.col("hour").is_between(8, 16) & (pl.col("inverter_id") != "A")
    df = df.with_columns(pl.when(trickle).then(0.5).otherwise(pl.col("inv_power")).alias("inv_power"))
    assert mark_outage_hours(df, CFG)["is_outage_hour"].sum() == 0


def test_single_inverter_outage_counts_core_hours_only():
    df = mark_outage_hours(plant(days=2, outages=[("A", 1, 6, 18)]), CFG)
    hours = df.filter(pl.col("is_outage_hour"))
    assert set(hours["inverter_id"]) == {"A"}
    assert sorted(hours["hour"]) == list(range(8, 17))


def test_short_blips_are_dropped_and_multiday_outages_merge():
    df = plant(days=5, outages=[("A", 1, 10, 11), ("B", 2, 8, 16), ("B", 3, 8, 16)])
    episodes = find_episodes(mark_outage_hours(df, CFG), CFG)
    assert episodes["inverter_id"].to_list() == ["B"]
    assert episodes["outage_hours"].to_list() == [18]
    assert episodes["start"][0] == START + timedelta(days=2, hours=8)


def test_target_window_and_masking():
    df, episodes = labelled(plant(days=40, outages=[("A", 20, 8, 16)]))
    onset = START + timedelta(days=20, hours=8)
    a = df.filter(pl.col("inverter_id") == "A")

    def row(ts):
        return a.filter(pl.col("timestamp") == ts).row(0, named=True)

    # 1-10 days ahead -> positive and usable
    assert row(onset - timedelta(days=5))["target"] == 1
    assert row(onset - timedelta(days=5))["label_ok"]
    # more than 10 days ahead -> negative
    assert row(onset - timedelta(days=11))["target"] == 0
    # less than a day ahead -> too late to act, excluded
    assert not row(onset - timedelta(hours=12))["label_ok"]
    # during the episode -> excluded, the onset itself is not "in the future"
    assert not row(onset)["label_ok"]
    assert not row(onset + timedelta(hours=4))["label_ok"]
    # healthy inverters never get positives
    assert df.filter(pl.col("inverter_id") != "A")["target"].sum() == 0


def test_rows_without_observed_future_are_excluded():
    df, _ = labelled(plant(days=40))
    end = df["timestamp"].max()
    tail = df.filter(pl.col("timestamp") > end - timedelta(days=CFG.lead_max_days))
    assert not tail["label_ok"].any()


def test_labels_ignore_features_after_the_row():
    """Changing data after a row's label window must not change that row's label."""
    base, _ = labelled(plant(days=40, outages=[("A", 20, 8, 16)]))
    changed, _ = labelled(plant(days=40, outages=[("A", 20, 8, 16), ("A", 35, 8, 16)]))
    cutoff = START + timedelta(days=24)
    cols = ["inverter_id", "timestamp", "target", "label_ok"]
    early = lambda d: d.filter(pl.col("timestamp") < cutoff).select(cols).sort("inverter_id", "timestamp")
    assert early(base).equals(early(changed))


def test_time_split_is_chronological_and_purged():
    df = plant(days=100)
    train, val, test, test_start = time_split(df, lead_max_days=10)
    assert train["timestamp"].max() + timedelta(days=10) <= val["timestamp"].min()
    assert val["timestamp"].max() + timedelta(days=10) <= test["timestamp"].min()
    assert test["timestamp"].min() == test_start


@pytest.mark.parametrize("lead", [(7, 10), (1, 10)])
def test_positive_rows_are_within_configured_lead(lead):
    cfg = LabelConfig(lead_min_days=lead[0], lead_max_days=lead[1])
    df, _ = labelled(plant(days=40, outages=[("A", 25, 8, 16)]), cfg)
    days = df.filter(pl.col("target") == 1)["days_to_onset"]
    assert days.min() >= lead[0] and days.max() <= lead[1]
