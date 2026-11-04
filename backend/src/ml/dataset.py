"""Build the hourly, per-inverter training table from raw data-logger exports.

Each raw CSV holds several inverters as wide `inverters[i].<field>` / `smu[i].<field>`
columns sampled every ~5 minutes. Two files from the same plant both number their
inverters from 0, so every inverter gets a plant-unique `inverter_idx` here
(file offset + local index) instead of being merged with its namesake.

Files are scanned lazily one inverter at a time, so a 450 MB export never has to fit
in memory at once.

Note: logger timestamps are local plant time even though the ISO strings end in `Z`
(power peaks at 11:00-12:00), so hours are used as-is.
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from src.ml.preprocess import add_all_features

# File-name prefix -> plant, matching the original Kaggle dataset layout
PLANT_PREFIXES = {
    "ICR2-": "plant_1",
    "80-1F-12-0F-AC-": "plant_2",
    "54-10-EC-8C-14-": "plant_3",
}

INVERTER_FIELDS = [
    "power", "temp", "pv1_power", "kwh_today", "kwh_total",
    "op_state", "alarm_code", "freq", "v_ab", "v_bc", "v_ca", "limit_percent",
]
# Status-like fields are aggregated with max (any alarm in the hour), the rest with mean
STATUS_FIELDS = {"op_state", "alarm_code"}


def plant_for_file(path: Path) -> str | None:
    name = path.name.removeprefix("Copy of ")
    for prefix, plant in PLANT_PREFIXES.items():
        if name.startswith(prefix):
            return plant
    return None


def discover_raw_files(raw_dir: Path) -> dict[str, list[Path]]:
    plants: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.raws.csv")):
        plant = plant_for_file(path)
        if plant:
            plants.setdefault(plant, []).append(path)
    return plants


def _logger_name(path: Path) -> str:
    return path.name.removeprefix("Copy of ").split(".raws")[0].removesuffix(".csv")


def _inverter_indices(columns: list[str]) -> list[int]:
    return sorted({int(m.group(1)) for c in columns if (m := re.match(r"inverters\[(\d+)\]\.", c))})


def load_inverter_hourly(path: Path, local_idx: int, columns: list[str]) -> pl.DataFrame:
    """Hourly aggregates for one inverter (plus its SMU string currents) from one file."""
    inv_prefix = f"inverters[{local_idx}]."
    smu_prefix = f"smu[{local_idx}]."

    rename = {}
    for c in columns:
        if c.startswith(inv_prefix) and c[len(inv_prefix):] in INVERTER_FIELDS:
            rename[c] = f"inv_{c[len(inv_prefix):]}"
        elif c.startswith(smu_prefix) and re.fullmatch(r"string\d+", c[len(smu_prefix):]):
            rename[c] = f"smu_{c[len(smu_prefix):]}"
    if "inv_power" not in rename.values():
        return pl.DataFrame()

    lf = (
        pl.scan_csv(path, infer_schema=False)
        .select(["timestamp", *rename])
        .rename(rename)
        .with_columns(
            pl.from_epoch(pl.col("timestamp").cast(pl.Int64, strict=False), time_unit="ms")
            .dt.truncate("1h").alias("timestamp"),
            # loggers write literal "NaN"; as null it is skipped by the hourly mean instead of poisoning it
            *[pl.col(c).cast(pl.Float64, strict=False).fill_nan(None) for c in rename.values()],
        )
        .filter(pl.col("timestamp").is_not_null())
    )
    aggs = [
        (pl.col(c).max() if c.removeprefix("inv_") in STATUS_FIELDS else pl.col(c).mean()).alias(c)
        for c in rename.values()
    ]
    return lf.group_by("timestamp").agg(aggs).sort("timestamp").collect()


def load_hourly(files: list[Path], plant: str, log=print) -> pl.DataFrame:
    """Hourly rows for every inverter in the given exports of one plant (no derived features)."""
    frames = []
    offset = 0
    for path in files:
        columns = pl.scan_csv(path, infer_schema=False).collect_schema().names()
        indices = _inverter_indices(columns)
        for local_idx in indices:
            hourly = load_inverter_hourly(path, local_idx, columns)
            if hourly.is_empty():
                continue
            frames.append(hourly.with_columns(
                pl.lit(offset + local_idx).cast(pl.Int32).alias("inverter_idx"),
                pl.lit(f"{_logger_name(path)}#{local_idx}").alias("inverter_id"),
            ))
            log(f"  {plant} {path.name[:40]} inverter {local_idx}: {hourly.height:,} hours")
        offset += max(indices, default=-1) + 1

    if not frames:
        raise ValueError("No inverters[i].power columns found; is this a raw data-logger export?")
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.with_columns(pl.lit(plant).alias("plant_id"), pl.col("timestamp").dt.hour().alias("hour"))


def build_plant_table(plant: str, files: list[Path], log=print) -> pl.DataFrame:
    return add_all_features(load_hourly(files, plant, log=log))


def load_dataset(raw_dir: Path, cache_dir: Path, rebuild: bool = False, log=print) -> pl.DataFrame:
    """All plants found in raw_dir, featurised, cached as one parquet per plant."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for plant, files in discover_raw_files(raw_dir).items():
        cache = cache_dir / f"{plant}_hourly.parquet"
        if cache.exists() and not rebuild:
            log(f"Loading cached {cache.name}")
            tables.append(pl.read_parquet(cache))
            continue
        log(f"Building {plant} from {len(files)} file(s)")
        table = build_plant_table(plant, files, log=log)
        table.write_parquet(cache, compression="zstd")
        tables.append(table)
    if not tables:
        raise FileNotFoundError(f"No *.raws.csv files for known plants in {raw_dir}")
    return pl.concat(tables, how="diagonal_relaxed")
