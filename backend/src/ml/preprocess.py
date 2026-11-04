import polars as pl
import numpy as np
import re
import gc

METADATA_DROPS = ["_id", "__v", "mac", "fromServer", "dataLoggerModelId",
                  "createdAt", "grid_master", "batteries"]
NULL_THRESHOLD = 0.90
MAX_FFILL_GAP = 6
DAYTIME_START = 6
DAYTIME_END = 18

def parse_column_schema(columns: list[str]) -> dict:
    pattern = re.compile(r"^(\w+)\[(\d+)\]\.(.+)$")
    subsystems = {}
    meta_cols = []
    for col in columns:
        match = pattern.match(col)
        if match:
            subsystem, idx, field = match.group(1), int(match.group(2)), match.group(3)
            subsystems.setdefault(subsystem, {}).setdefault(idx, []).append(field)
        else:
            meta_cols.append(col)
    return {"subsystems": subsystems, "meta": meta_cols}

def unpivot_inverters(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    subsystems = schema["subsystems"]
    if "inverters" not in subsystems:
        return df
    inverter_indices = sorted(subsystems["inverters"].keys())
    if len(inverter_indices) <= 1:
        rename_map = {}
        for col in df.columns:
            m = re.match(r"^inverters\[0\]\.(.+)$", col)
            if m:
                rename_map[col] = f"inv_{m.group(1)}"
        if rename_map:
            df = df.rename(rename_map)
            df = df.with_columns(pl.lit(0).cast(pl.Int16).alias("inverter_idx"))
        return df

    common_fields = set(subsystems["inverters"][inverter_indices[0]])
    for idx in inverter_indices[1:]:
        common_fields &= set(subsystems["inverters"][idx])

    inverter_col_pattern = re.compile(r"^inverters\[\d+\]\.")
    shared_cols = [c for c in df.columns if not inverter_col_pattern.match(c)]

    inverter_dfs = []
    for idx in inverter_indices:
        inv_cols = {}
        for field in common_fields:
            src = f"inverters[{idx}].{field}"
            if src in df.columns:
                inv_cols[src] = f"inv_{field}"
        if not inv_cols:
            continue
        sub = df.select(shared_cols + list(inv_cols.keys()))
        sub = sub.rename(inv_cols)
        sub = sub.with_columns(pl.lit(idx).cast(pl.Int16).alias("inverter_idx"))
        inverter_dfs.append(sub)

    if inverter_dfs:
        df_long = pl.concat(inverter_dfs, how="diagonal_relaxed")
        return df_long
    return df

def handle_smu(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    subsystems = schema["subsystems"]
    if "smu" not in subsystems:
        return df

    smu_indices = sorted(subsystems["smu"].keys())
    common_smu = set(subsystems["smu"][smu_indices[0]])
    for idx in smu_indices[1:]:
        common_smu &= set(subsystems["smu"][idx])

    existing_smu = [c for c in df.columns if c.startswith("smu[")]
    if not existing_smu or "inverter_idx" not in df.columns:
        rename_map = {}
        for col in df.columns:
            m = re.match(r"^smu\[\d+\]\.(.+)$", col)
            if m:
                target = f"smu_{m.group(1)}"
                if target not in rename_map.values():
                    rename_map[col] = target
        if rename_map:
            df = df.rename(rename_map)
        return df

    for field in common_smu:
        target = f"smu_{field}"
        expr = pl.lit(None).cast(pl.Float64)
        for idx in smu_indices:
            src = f"smu[{idx}].{field}"
            if src in df.columns:
                expr = pl.when(pl.col("inverter_idx") == idx).then(pl.col(src)).otherwise(expr)
        df = df.with_columns(expr.alias(target))

    smu_originals = [c for c in df.columns if c.startswith("smu[")]
    if smu_originals:
        df = df.drop(smu_originals)

    return df

def flatten_remaining(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    for col in df.columns:
        m = re.match(r"^(\w+)\[(\d+)\]\.(.+)$", col)
        if m:
            subsys = m.group(1).rstrip("s") if m.group(1).endswith("s") else m.group(1)
            new_name = f"{subsys}_{m.group(3)}"
            if new_name in rename_map.values() or new_name in df.columns:
                new_name = f"{subsys}{m.group(2)}_{m.group(3)}"
            rename_map[col] = new_name
    if rename_map:
        df = df.rename(rename_map)
    return df

def process_single_csv(filepath: str) -> pl.DataFrame:
    df = pl.read_csv(
        filepath,
        infer_schema_length=10_000,
        ignore_errors=True,
        truncate_ragged_lines=True,
        null_values=["", "NA", "N/A", "null", "NULL", "None", "nan",
                     "NaN", "-", "--", "?", "#N/A", "#VALUE!", "undefined"],
    )

    schema = parse_column_schema(df.columns)

    drops = [c for c in df.columns if any(m.lower() in c.lower() for m in METADATA_DROPS)]
    drops = [c for c in drops if c != "timestamp"]
    if drops:
        df = df.drop([c for c in drops if c in df.columns])

    null_drops = [c for c in df.columns if df[c].null_count() / df.height > NULL_THRESHOLD]
    if null_drops:
        df = df.drop(null_drops)

    if "timestamp" in df.columns and df["timestamp"].dtype == pl.String:
        df = df.with_columns(pl.col("timestamp").str.to_datetime(strict=False))
    if "timestampDate" in df.columns:
        if "timestamp" not in df.columns or df["timestamp"].null_count() == df.height:
            if df["timestampDate"].dtype == pl.String:
                df = df.with_columns(pl.col("timestampDate").str.to_datetime(strict=False).alias("timestamp"))
        df = df.drop("timestampDate")

    const_cols = [c for c in df.columns
                  if df[c].drop_nulls().n_unique() <= 1
                  and c not in ("timestamp",)]
    if const_cols:
        df = df.drop(const_cols)

    id_cols = [c for c in df.columns if re.search(r"\.(model|serial|id)$", c)]
    if id_cols:
        df = df.drop(id_cols)

    schema = parse_column_schema(df.columns)
    df = unpivot_inverters(df, schema)
    df = handle_smu(df, schema)
    df = flatten_remaining(df)

    for col_name in df.columns:
        if df[col_name].dtype != pl.String:
            continue
        sample = df[col_name].drop_nulls().head(100)
        if sample.len() == 0:
            continue
        try:
            casted = sample.cast(pl.Float64, strict=False)
            if 1.0 - (casted.null_count() / sample.len()) > 0.8:
                df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))
        except Exception:
            pass
    return df

def resample_hourly(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" not in df.columns:
        return df
        
    if df["timestamp"].dtype != pl.Datetime:
        if df["timestamp"].dtype in (pl.Float64, pl.Float32, pl.Int64):
            sample_val = df["timestamp"].drop_nulls().head(1).item()
            if sample_val > 1e12:
                df = df.with_columns(
                    pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="ms").alias("timestamp")
                )
            else:
                df = df.with_columns(
                    pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="s").alias("timestamp")
                )

    group_cols = {"timestamp", "inverter_idx"}

    categorical_patterns = re.compile(r"(op_state|alarm_code|alarm|fault|status|flag)", re.I)
    categorical_cols = [c for c in df.columns
                        if categorical_patterns.search(c)
                        and c not in group_cols
                        and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32,
                                            pl.Int16, pl.Int8, pl.UInt32)]

    numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32)
    numeric_cols = [c for c in df.columns
                    if df[c].dtype in numeric_types
                    and c not in group_cols
                    and c not in categorical_cols]

    drop_cols = [c for c in df.columns
                 if c not in group_cols
                 and c not in numeric_cols
                 and c not in categorical_cols
                 and c != "timestamp"]
    if drop_cols:
        df = df.drop(drop_cols)

    df = df.sort(["inverter_idx", "timestamp"])

    agg_exprs = []
    for c in categorical_cols:
        agg_exprs.append(
            pl.col(c).drop_nulls().mode().first().alias(c)
        )

    for c in numeric_cols:
        agg_exprs.append(pl.col(c).mean().alias(c))

    df = df.group_by_dynamic(
        "timestamp", every="1h", group_by="inverter_idx"
    ).agg(agg_exprs)

    return df


def impute_missing(df: pl.DataFrame) -> pl.DataFrame:
    alarm_pattern = re.compile(r"(alarm|fault|error|warning|flag|status|op_state)", re.I)

    numeric_cols = [c for c in df.columns if df[c].dtype in (
        pl.Float32, pl.Float64, pl.Int64, pl.Int32, pl.Int16, pl.Int8)]

    alarm_cols = [c for c in numeric_cols if alarm_pattern.search(c)]
    telemetry_cols = [c for c in numeric_cols if c not in alarm_cols]

    has_group = "inverter_idx" in df.columns

    if telemetry_cols:
        if has_group:
            df = df.with_columns([
                pl.col(c).forward_fill(limit=MAX_FFILL_GAP).over("inverter_idx")
                for c in telemetry_cols
            ])
            df = df.with_columns([
                pl.col(c).backward_fill(limit=2).over("inverter_idx")
                for c in telemetry_cols
            ])
        else:
            df = df.with_columns([pl.col(c).forward_fill(limit=MAX_FFILL_GAP) for c in telemetry_cols])
            df = df.with_columns([pl.col(c).backward_fill(limit=2) for c in telemetry_cols])

    if alarm_cols:
        df = df.with_columns([pl.col(c).fill_null(0) for c in alarm_cols])

    return df


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" not in df.columns or df["timestamp"].dtype not in (pl.Datetime,):
        return df
    if "hour" not in df.columns:
        df = df.with_columns(pl.col("timestamp").dt.hour().alias("hour"))

    df = df.with_columns([
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        pl.col("timestamp").dt.month().alias("month"),
        ((pl.col("hour") >= DAYTIME_START) & (pl.col("hour") <= DAYTIME_END))
        .cast(pl.Int8).alias("is_daytime"),
    ])
    return df

def add_rolling_telemetry_features(df: pl.DataFrame) -> pl.DataFrame:
    has_group = "inverter_idx" in df.columns

    key_signals = [s for s in ["inv_power", "inv_temp", "inv_freq", "inv_pv1_power",
                                "inv_kwh_today", "inv_v_ab", "inv_v_bc", "inv_v_ca"]
                   if s in df.columns]
    if not key_signals:
        return df

    windows = {"3d": 72, "7d": 168}

    for signal in key_signals:
        sig_safe = signal.replace("inv_", "")
        for label, w in windows.items():
            mean_expr = pl.col(signal).rolling_mean(window_size=w, min_periods=1)
            std_expr = pl.col(signal).rolling_std(window_size=w, min_periods=2)
            if has_group:
                mean_expr = mean_expr.over("inverter_idx")
                std_expr = std_expr.over("inverter_idx")
            try:
                df = df.with_columns([
                    mean_expr.alias(f"roll_{sig_safe}_mean_{label}"),
                    std_expr.alias(f"roll_{sig_safe}_std_{label}"),
                ])
            except Exception:
                pass

    return df

def add_degradation_features(df: pl.DataFrame) -> pl.DataFrame:
    has_group = "inverter_idx" in df.columns
    new_cols = []
    temp_cols = []

    windows = {"7d": 168, "14d": 336}

    for signal, label in [("inv_power", "power"), ("inv_kwh_today", "kwh"), ("inv_temp", "temp")]:
        if signal not in df.columns:
            continue
        for wlabel, wsize in windows.items():
            rmean_col = f"_{label}_rmean_{wlabel}"
            expr = pl.col(signal).rolling_mean(window_size=wsize, min_periods=1)
            if has_group:
                expr = expr.over("inverter_idx")
            try:
                df = df.with_columns(expr.alias(rmean_col))
                temp_cols.append(rmean_col)

                if label == "temp":
                    new_cols.append((pl.col(signal) - pl.col(rmean_col)).alias(f"degrad_{label}_dev_{wlabel}"))
                else:
                    new_cols.append(
                        (pl.col(signal) / (pl.col(rmean_col) + 1e-6)).alias(f"degrad_{label}_ratio_{wlabel}")
                    )
            except Exception:
                pass

    if new_cols:
        df = df.with_columns(new_cols)

    temp_cols = [c for c in temp_cols if c in df.columns]
    if temp_cols:
        df = df.drop(temp_cols)

    return df

def add_cumulative_stress_features(df: pl.DataFrame) -> pl.DataFrame:
    has_group = "inverter_idx" in df.columns

    if "inv_temp" in df.columns:
        df = df.with_columns((pl.col("inv_temp") > 55).cast(pl.Int8).alias("_temp_stress"))
        expr = pl.col("_temp_stress").rolling_sum(window_size=168, min_periods=1)
        if has_group:
            expr = expr.over("inverter_idx")
        df = df.with_columns(expr.alias("stress_hightemp_7d"))
        df = df.drop("_temp_stress")

    if "inv_power" in df.columns and "hour" in df.columns:
        df = df.with_columns(
            ((pl.col("inv_power").abs() < 0.1) &
             (pl.col("hour") >= DAYTIME_START) &
             (pl.col("hour") <= DAYTIME_END))
            .cast(pl.Int8).alias("_dz")
        )

        for label, w in {"1d": 24, "3d": 72}.items():
            expr = pl.col("_dz").rolling_sum(window_size=w, min_periods=1)
            if has_group:
                expr = expr.over("inverter_idx")
            try:
                df = df.with_columns(expr.alias(f"stress_zero_power_{label}"))
            except Exception:
                pass

        df = df.drop("_dz")

    if "is_failure_event" in df.columns:
        for label, w in {"3d": 72, "7d": 168}.items():
            expr = pl.col("is_failure_event").rolling_sum(window_size=w, min_periods=1)
            if has_group:
                expr = expr.over("inverter_idx")
            try:
                df = df.with_columns(expr.alias(f"stress_failure_count_{label}"))
            except Exception:
                pass

    return df

def add_nighttime_anomaly_features(df: pl.DataFrame) -> pl.DataFrame:
    if "hour" not in df.columns:
        return df

    has_group = "inverter_idx" in df.columns
    night = (pl.col("hour") < DAYTIME_START) | (pl.col("hour") > DAYTIME_END)
    new_cols = []

    if "inv_power" in df.columns:
        new_cols.append(
            pl.when(night).then((pl.col("inv_power").abs() > 0.5).cast(pl.Int8))
            .otherwise(0).alias("anom_night_power")
        )
    if "inv_temp" in df.columns:
        new_cols.append(
            pl.when(night).then((pl.col("inv_temp") > 40).cast(pl.Int8))
            .otherwise(0).alias("anom_night_hightemp")
        )

    if new_cols:
        df = df.with_columns(new_cols)
        for ac in ["anom_night_power", "anom_night_hightemp"]:
            if ac in df.columns:
                expr = pl.col(ac).rolling_sum(window_size=168, min_periods=1)
                if has_group:
                    expr = expr.over("inverter_idx")
                try:
                    df = df.with_columns(expr.alias(f"{ac}_7d"))
                except Exception:
                    pass
    return df

def add_string_health_features(df: pl.DataFrame) -> pl.DataFrame:
    has_group = "inverter_idx" in df.columns

    smu_cols = sorted([c for c in df.columns if re.match(r"smu_string\d+$", c)])
    if len(smu_cols) < 3:
        return df

    df = df.with_columns([
        pl.mean_horizontal(smu_cols).alias("str_mean"),
        pl.min_horizontal(smu_cols).alias("str_min"),
        pl.max_horizontal(smu_cols).alias("str_max"),
        (pl.max_horizontal(smu_cols) - pl.min_horizontal(smu_cols)).alias("str_spread"),
        (pl.min_horizontal(smu_cols) / (pl.max_horizontal(smu_cols) + 1e-6)).alias("str_worst_ratio"),
    ])

    row_mean = pl.mean_horizontal(smu_cols)
    underperf = [((pl.col(c) < row_mean * 0.5) & (pl.col(c).is_not_null())).cast(pl.Int8) for c in smu_cols]
    df = df.with_columns(pl.sum_horizontal(underperf).alias("str_underperf_count"))

    for label, w in {"3d": 72, "7d": 168}.items():
        expr = pl.col("str_mean").rolling_mean(window_size=w, min_periods=1)
        if has_group:
            expr = expr.over("inverter_idx")
        df = df.with_columns(expr.alias(f"str_mean_rmean_{label}"))

    df = df.with_columns(
        (pl.col("str_mean") / (pl.col("str_mean_rmean_7d") + 1e-6)).alias("str_trend_7d")
    )

    expr = pl.col("str_worst_ratio").rolling_mean(window_size=168, min_periods=1)
    if has_group:
        expr = expr.over("inverter_idx")
    df = df.with_columns(expr.alias("str_worst_ratio_rmean_7d"))

    return df


def add_inter_inverter_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compare each inverter against its plant peers at the same timestamp (call per plant)."""
    if "inverter_idx" not in df.columns or df["inverter_idx"].n_unique() < 2:
        return df

    new_cols = []
    temp_cols = []

    for signal, name in [("inv_power", "power"), ("inv_temp", "temp")]:
        if signal not in df.columns:
            continue

        med = f"_plant_med_{name}"
        std = f"_plant_std_{name}"
        df = df.with_columns([
            pl.col(signal).median().over("timestamp").alias(med),
            pl.col(signal).std().over("timestamp").alias(std),
        ])
        temp_cols.extend([med, std])

        new_cols.append((pl.col(signal) - pl.col(med)).alias(f"peer_{name}_dev"))
        new_cols.append(
            ((pl.col(signal) - pl.col(med)) / (pl.col(std) + 1e-6)).alias(f"peer_{name}_zscore")
        )

    if "inv_power" in df.columns:
        df = df.with_columns(pl.col("inv_power").max().over("timestamp").alias("_pmax"))
        temp_cols.append("_pmax")
        new_cols.append((pl.col("inv_power") / (pl.col("_pmax") + 1e-6)).alias("peer_power_vs_best"))

    if new_cols:
        df = df.with_columns(new_cols)

    temp_cols = [c for c in temp_cols if c in df.columns]
    if temp_cols:
        df = df.drop(temp_cols)

    return df


def add_all_features(df: pl.DataFrame) -> pl.DataFrame:
    """Feature engineering shared by training, evaluation and raw-upload inference.

    Expects hourly rows for one plant, with `inverter_idx` and `timestamp`.
    """
    df = df.sort(["inverter_idx", "timestamp"])
    df = add_temporal_features(df)
    df = add_rolling_telemetry_features(df)
    df = add_degradation_features(df)
    df = add_cumulative_stress_features(df)
    df = add_nighttime_anomaly_features(df)
    df = add_string_health_features(df)
    df = add_inter_inverter_features(df)
    return df
