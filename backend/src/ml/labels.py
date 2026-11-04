"""Failure labels for inverter-hour rows.

The original label ("any hour between 06:00 and 18:00 with zero power") fired on
sunrise/sunset hours and plant-wide outages, so most inverters "failed" every few days.

Here an *outage hour* is a core-daylight hour where the inverter produces nothing while
most of its plant peers are producing (so grid trips and cloudy days don't count). An
*outage episode* is a run of at least `min_hours` outage hours; runs separated by less
than `merge_gap_hours` (e.g. the night between two dead days) are one episode. The
prediction target for a row is "an episode starts between lead_min and lead_max days
from now".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import polars as pl

KEY = "inverter_id"


@dataclass(frozen=True)
class LabelConfig:
    core_start_hour: int = 8
    core_end_hour: int = 16
    zero_power: float = 0.1
    # a peer counts as producing above this share of its own capacity (p99.5 output)
    peer_producing_pu: float = 0.05
    peer_producing_frac: float = 0.5
    min_hours: int = 3
    merge_gap_hours: int = 24
    lead_min_days: float = 1.0
    lead_max_days: float = 10.0


def mark_outage_hours(df: pl.DataFrame, cfg: LabelConfig) -> pl.DataFrame:
    """Add `is_outage_hour`: core daylight, inverter at zero, most plant peers producing.

    "Producing" is relative to each peer's capacity, so a heavily overcast or curtailed
    hour where everyone trickles a few watts doesn't make one zero look like a fault.
    """
    # Capacity as its own column: a window expression nested inside another isn't evaluated per group
    df = df.with_columns(pl.col("inv_power").quantile(0.995).over(KEY).alias("_capacity"))
    producing = ((pl.col("inv_power").abs() >= cfg.zero_power)
                 & (pl.col("inv_power") >= cfg.peer_producing_pu * pl.col("_capacity"))).cast(pl.Int32)
    df = df.with_columns(
        producing.sum().over(["plant_id", "timestamp"]).alias("_n_producing"),
        pl.len().over(["plant_id", "timestamp"]).alias("_n_reporting"),
    )
    is_zero = pl.col("inv_power").abs() < cfg.zero_power
    # This inverter is at zero, so every producing inverter at this timestamp is a peer
    peers = pl.col("_n_reporting") - 1
    peers_ok = (peers >= 1) & (pl.col("_n_producing") >= cfg.peer_producing_frac * peers)
    core = pl.col("hour").is_between(cfg.core_start_hour, cfg.core_end_hour)
    return df.with_columns(
        (is_zero & core & peers_ok).fill_null(False).alias("is_outage_hour")
    ).drop(["_n_producing", "_n_reporting", "_capacity"])


def find_episodes(df: pl.DataFrame, cfg: LabelConfig) -> pl.DataFrame:
    """One row per outage episode: inverter_id, start, end, outage_hours."""
    hours = df.filter(pl.col("is_outage_hour")).select(KEY, "timestamp").sort(KEY, "timestamp")
    if hours.is_empty():
        return pl.DataFrame(schema={KEY: pl.String, "start": pl.Datetime("ms"),
                                    "end": pl.Datetime("ms"), "outage_hours": pl.UInt32})
    gap = pl.col("timestamp").diff().over(KEY)
    new_episode = gap.is_null() | (gap > timedelta(hours=cfg.merge_gap_hours))
    return (
        hours.with_columns(new_episode.cast(pl.Int32).cum_sum().over(KEY).alias("_ep"))
        .group_by(KEY, "_ep")
        .agg(pl.col("timestamp").min().alias("start"),
             pl.col("timestamp").max().alias("end"),
             pl.len().alias("outage_hours"))
        .filter(pl.col("outage_hours") >= cfg.min_hours)
        .drop("_ep")
        .sort(KEY, "start")
    )


def add_targets(df: pl.DataFrame, episodes: pl.DataFrame, cfg: LabelConfig) -> pl.DataFrame:
    """Add `days_to_onset`, `target` and `label_ok`.

    `label_ok` is False for rows that shouldn't be trained or scored on: during an
    episode, less than lead_min before one (too late to act), or so close to the end
    of the data that the next lead_max days aren't observed.
    """
    df = df.sort(KEY, "timestamp")
    ep = episodes.select(KEY, pl.col("start").cast(df.schema["timestamp"]),
                         pl.col("end").cast(df.schema["timestamp"])).sort(KEY, "start")

    nxt = ep.select(KEY, pl.col("start").alias("_next_onset"), pl.col("start").alias("_join"))
    prev = ep.select(KEY, pl.col("end").alias("_prev_end"), pl.col("start").alias("_join"))

    df = (
        df.with_columns(pl.col("timestamp").alias("_join"))
        # strictly after: an episode starting this hour is "now", not the future
        .join_asof(nxt.with_columns(pl.col("_join") - timedelta(seconds=1)),
                   on="_join", by=KEY, strategy="forward", check_sortedness=False)
        .join_asof(prev, on="_join", by=KEY, strategy="backward", check_sortedness=False)
        .drop("_join")
    )

    data_end = pl.col("timestamp").max().over(KEY)
    days = (pl.col("_next_onset") - pl.col("timestamp")).dt.total_seconds() / 86400
    in_episode = pl.col("_prev_end").is_not_null() & (pl.col("timestamp") <= pl.col("_prev_end"))
    observed = pl.col("timestamp") <= data_end - timedelta(days=cfg.lead_max_days)
    too_late = days.is_not_null() & (days < cfg.lead_min_days)
    positive = days.is_not_null() & (days >= cfg.lead_min_days) & (days <= cfg.lead_max_days)

    return df.with_columns(
        days.alias("days_to_onset"),
        positive.cast(pl.Int8).alias("target"),
        (~in_episode & ~too_late & (observed | positive)).alias("label_ok"),
    ).drop(["_next_onset", "_prev_end"])


def legacy_targets(df: pl.DataFrame, window=(7.0, 10.0)) -> pl.DataFrame:
    """The hackathon label: any 06-18h zero-power hour, target if the next one is 7-10 days out."""
    df = df.sort(KEY, "timestamp").with_columns(
        ((pl.col("inv_power").abs() < 0.1) & pl.col("hour").is_between(6, 18)).alias("_fail")
    )
    next_fail = (pl.when(pl.col("_fail")).then(pl.col("timestamp"))
                 .backward_fill().over(KEY))
    days = (next_fail - pl.col("timestamp")).dt.total_seconds() / 86400
    return df.with_columns(
        days.is_between(*window).fill_null(False).cast(pl.Int8).alias("legacy_target")
    ).drop("_fail")
