"""Same-day detection of inverter-specific underperformance.

Instead of forecasting failures days ahead (which the data doesn't support, see
reports/evaluation.md), each inverter is compared with what it *should* be producing
given what its plant peers produce right now:

1. Normalise power to per-unit of the inverter's own capacity (p99.5 of its output in
   a reference period), so inverters of different sizes are comparable.
2. Describe current conditions from the *other* inverters in the plant (median and
   spread of their per-unit output, their temperature), so the inverter's own fault
   can't leak into its expectation.
3. An XGBoost regressor maps conditions -> expected per-unit power, plus the
   inverter's usual ratio to its peers during the reference period.
4. A one-sided CUSUM over the relative shortfall raises an alarm when the inverter
   keeps producing less than expected.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
import xgboost as xgb

KEY = "inverter_id"
CORE_HOURS = (8, 16)
MODEL_FEATURES = [
    "peer_median_pu", "peer_q25_pu", "peer_q75_pu", "peer_median_temp",
    "hour", "doy_sin", "doy_cos", "ref_ratio",
]


@dataclass(frozen=True)
class DetectorConfig:
    k: float = 0.1          # shortfall tolerated per hour before evidence accumulates (10%)
    h: float = 2.0          # alarm when accumulated shortfall exceeds this
    min_expected_pu: float = 0.1  # ignore hours with too little sun to judge
    min_peers: int = 2


# --------------------------------------------------------------------------- context

def capacities(df: pl.DataFrame, until: datetime | None = None) -> dict[str, float]:
    ref = df if until is None else df.filter(pl.col("timestamp") < until)
    caps = ref.group_by(KEY).agg(pl.col("inv_power").quantile(0.995).alias("cap"))
    return {k: max(float(v or 0.0), 1e-6) for k, v in caps.iter_rows()}


def add_per_unit(df: pl.DataFrame, caps: dict[str, float]) -> pl.DataFrame:
    cap = pl.col(KEY).replace_strict(caps, default=None, return_dtype=pl.Float64)
    return df.with_columns((pl.col("inv_power") / cap).fill_nan(None).clip(0.0, 1.5).alias("pu"))


def add_peer_context(df: pl.DataFrame) -> pl.DataFrame:
    """Per-row stats of the *other* inverters in the same plant at the same timestamp."""
    out = []
    for (plant,), part in df.group_by("plant_id"):
        wide_pu = part.pivot(index="timestamp", on=KEY, values="pu").sort("timestamp")
        inverters = [c for c in wide_pu.columns if c != "timestamp"]
        pu = wide_pu.select(inverters).to_numpy().astype(float)
        has_temp = "inv_temp" in part.columns
        if has_temp:
            temp = (part.pivot(index="timestamp", on=KEY, values="inv_temp").sort("timestamp")
                    .select(inverters).to_numpy().astype(float))
        cols = {k: [] for k in ["peer_median_pu", "peer_q25_pu", "peer_q75_pu", "peer_count", "peer_median_temp"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            for j in range(len(inverters)):
                others = np.delete(pu, j, axis=1)
                q25, med, q75 = np.nanpercentile(others, [25, 50, 75], axis=1) if others.shape[1] else (np.full(len(pu), np.nan),) * 3
                cols["peer_median_pu"].append(med)
                cols["peer_q25_pu"].append(q25)
                cols["peer_q75_pu"].append(q75)
                cols["peer_count"].append((~np.isnan(others)).sum(axis=1))
                cols["peer_median_temp"].append(
                    np.nanmedian(np.delete(temp, j, axis=1), axis=1) if has_temp and others.shape[1] else np.full(len(pu), np.nan))
        n = len(wide_pu)
        long = pl.DataFrame({
            "timestamp": np.tile(wide_pu["timestamp"].to_numpy(), len(inverters)),
            KEY: np.repeat(inverters, n),
            **{k: np.concatenate(v) for k, v in cols.items()},
        }).with_columns(pl.col("timestamp").cast(part.schema["timestamp"]), pl.lit(plant).alias("plant_id"),
                        pl.col("^peer_.*_(pu|temp)$").fill_nan(None))
        out.append(long)
    ctx = pl.concat(out)
    doy = pl.col("timestamp").dt.ordinal_day() * (2 * np.pi / 365.25)
    return (df.join(ctx, on=["plant_id", KEY, "timestamp"], how="left")
              .with_columns(doy.sin().alias("doy_sin"), doy.cos().alias("doy_cos")))


def reference_ratios(df: pl.DataFrame, until: datetime | None = None) -> dict[str, float]:
    """Each inverter's usual output relative to its peers (shading, orientation, string count)."""
    ref = df if until is None else df.filter(pl.col("timestamp") < until)
    ref = ref.filter(core_mask() & (pl.col("peer_median_pu") > 0.2) & (pl.col("pu") > 0))
    ratios = ref.group_by(KEY).agg((pl.col("pu") / pl.col("peer_median_pu")).median().alias("r"))
    return {k: float(v) for k, v in ratios.iter_rows()}


def add_reference_ratio(df: pl.DataFrame, ratios: dict[str, float]) -> pl.DataFrame:
    return df.with_columns(
        pl.col(KEY).replace_strict(ratios, default=1.0, return_dtype=pl.Float64).alias("ref_ratio"))


def core_mask() -> pl.Expr:
    return pl.col("hour").is_between(*CORE_HOURS)


def prepare_frame(df: pl.DataFrame, reference_until: datetime | None = None) -> pl.DataFrame:
    """Per-unit power, peer context and reference ratios; reference stats use rows before reference_until."""
    df = add_per_unit(df, capacities(df, reference_until))
    df = add_peer_context(df)
    return add_reference_ratio(df, reference_ratios(df, reference_until))


# --------------------------------------------------------------------------- expected power

def fit_expected_power(train: pl.DataFrame, val: pl.DataFrame | None = None) -> xgb.XGBRegressor:
    """Learn normal behaviour from healthy rows (known outage hours excluded); MAE loss for robustness."""
    def usable(d):
        d = d.filter(core_mask() & (pl.col("peer_count") >= 1) & pl.col("pu").is_not_null()
                     & pl.col("peer_median_pu").is_not_null())
        if "is_outage_hour" in d.columns:
            d = d.filter(~pl.col("is_outage_hour"))
        return d.select(MODEL_FEATURES).to_numpy().astype("float32"), d["pu"].to_numpy()

    X, y = usable(train)
    model = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=600, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1, tree_method="hist",
                             early_stopping_rounds=50 if val is not None else None)
    eval_set = [usable(val)] if val is not None else None
    model.fit(X, y, eval_set=eval_set, verbose=False)
    return model


def add_expected(df: pl.DataFrame, model: xgb.XGBRegressor | None) -> pl.DataFrame:
    """`expected_pu` from the model, or from peers x reference ratio when model is None (baseline)."""
    if model is None:
        expected = (pl.col("peer_median_pu") * pl.col("ref_ratio")).clip(0.0, 1.5)
        return df.with_columns(expected.alias("expected_pu"))
    X = df.select(MODEL_FEATURES).to_numpy().astype("float32")
    return df.with_columns(pl.Series("expected_pu", np.clip(model.predict(X), 0.0, 1.5)))


def add_shortfall(df: pl.DataFrame, cfg: DetectorConfig) -> pl.DataFrame:
    """Relative shortfall per hour; null where the hour can't be judged (night, too few peers)."""
    judgeable = (core_mask() & (pl.col("expected_pu") >= cfg.min_expected_pu)
                 & (pl.col("peer_count") >= cfg.min_peers) & pl.col("pu").is_not_null())
    rel = ((pl.col("pu") - pl.col("expected_pu")) / pl.col("expected_pu")).clip(-1.5, 1.5)
    return df.with_columns(pl.when(judgeable).then(rel).otherwise(None).alias("residual"))


# --------------------------------------------------------------------------- CUSUM

def cusum(residuals: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided CUSUM on shortfall (-residual). Returns (statistic, alarm) arrays; resets after an alarm."""
    s = 0.0
    stat = np.zeros(len(residuals))
    alarm = np.zeros(len(residuals), dtype=bool)
    for i, r in enumerate(residuals):
        s = max(0.0, s + (-r - k))
        stat[i] = s
        if s > h:
            alarm[i] = True
            s = 0.0
    return stat, alarm


def run_detector(df: pl.DataFrame, cfg: DetectorConfig) -> pl.DataFrame:
    """Judgeable rows with `cusum` and `alarm` columns, per inverter in time order."""
    rows = df.filter(pl.col("residual").is_not_null()).sort(KEY, "timestamp")
    stats, alarms = [], []
    for _, part in rows.group_by(KEY, maintain_order=True):
        st, al = cusum(part["residual"].to_numpy(), cfg.k, cfg.h)
        stats.append(st)
        alarms.append(al)
    if not stats:
        return rows.with_columns(pl.lit(None, pl.Float64).alias("cusum"), pl.lit(False).alias("alarm"))
    return rows.with_columns(pl.Series("cusum", np.concatenate(stats)), pl.Series("alarm", np.concatenate(alarms)))


def daily_summary(scored: pl.DataFrame) -> pl.DataFrame:
    """Per inverter-day: energy vs expected and whether the detector alarmed."""
    return (scored.with_columns(pl.col("timestamp").dt.date().alias("day"))
            .group_by(KEY, "plant_id", "day")
            .agg((pl.col("pu").sum() / pl.col("expected_pu").sum()).alias("energy_ratio"),
                 pl.col("alarm").any(), pl.col("cusum").max().alias("max_cusum"), pl.len().alias("hours"))
            .sort(KEY, "day"))
