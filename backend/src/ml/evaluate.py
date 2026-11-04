"""Leak-free evaluation of the failure-prediction model.

Rebuilds the hourly dataset from raw logger exports, labels outage episodes
(see labels.py), and compares:

  A. hackathon setup   - legacy label, original 20 features + plant one-hot, random split
  B. A with a time split
  C. new label, original features minus leaky ones, time split
  D. new label, all engineered features + outage history, time split
  E. D trained on one plant, tested on the other(s)
  plus a rule baseline (outage hours in the last 30 days) on C/D's test set.

Usage (from backend/):
    python -m src.ml.evaluate --raw-dir <folder with *.raws.csv>
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from src.ml.dataset import load_dataset
from src.ml.labels import KEY, LabelConfig, add_targets, find_episodes, legacy_targets, mark_outage_hours

BACKEND_DIR = Path(__file__).resolve().parents[2]

ORIGINAL_FEATURES = [
    "inv_kwh_total", "roll_temp_mean_7d", "roll_kwh_today_std_7d",
    "roll_temp_std_7d", "roll_temp_std_3d", "roll_temp_mean_3d",
    "roll_pv1_power_std_7d", "anom_night_power_7d", "roll_kwh_today_mean_7d",
    "roll_kwh_today_std_3d", "str_worst_ratio_rmean_7d", "day_of_week",
    "inv_power", "str_mean_rmean_7d", "roll_kwh_today_mean_3d",
    "stress_hightemp_7d", "is_daytime", "anom_night_hightemp_7d",
    "roll_power_std_3d", "roll_pv1_power_mean_3d",
]
# Lifetime counter and calendar fields identify *when* (and which inverter) a row is,
# which a random split rewards and a time split punishes.
LEAKY = {"inv_kwh_total", "day_of_week", "month"}
NON_FEATURES = {
    "timestamp", "inverter_idx", KEY, "plant_id", "target", "legacy_target",
    "days_to_onset", "label_ok", "is_outage_hour",
}
HISTORY_WINDOWS = {"outage_hours_7d": 168, "outage_hours_30d": 720}
BASELINE_SCORE = "outage_hours_30d"

XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05, eval_metric="aucpr",
                  early_stopping_rounds=50, random_state=42, n_jobs=-1, tree_method="hist")


# --------------------------------------------------------------------------- data

def prepare(df: pl.DataFrame, cfg: LabelConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = mark_outage_hours(df, cfg).sort(KEY, "timestamp")
    df = df.with_columns([
        pl.col("is_outage_hour").cast(pl.Int32).rolling_sum(window_size=w, min_samples=1).over(KEY).alias(name)
        for name, w in HISTORY_WINDOWS.items()
    ])
    episodes = find_episodes(df, cfg)
    df = add_targets(df, episodes, cfg)
    df = legacy_targets(df)
    return df, episodes


def feature_sets(df: pl.DataFrame) -> dict[str, list[str]]:
    numeric = [c for c, t in df.schema.items() if t.is_numeric() or t == pl.Boolean]
    full = [c for c in numeric
            if c not in NON_FEATURES and c not in LEAKY and not c.startswith("smu_string")]
    return {
        "original": [c for c in ORIGINAL_FEATURES if c in df.columns],
        "original_leakfree": [c for c in ORIGINAL_FEATURES if c in df.columns and c not in LEAKY],
        "full": full,
    }


def to_xy(df: pl.DataFrame, features: list[str], target: str, one_hot_plant=False):
    X = df.select(features).to_pandas()
    if one_hot_plant:
        X = pd.concat([X, pd.get_dummies(df["plant_id"].to_pandas(), prefix="plant_id")], axis=1)
    return X.astype("float32"), df[target].to_numpy()


def time_split(df: pl.DataFrame, lead_max_days: float, val_q=0.6, test_q=0.8):
    """Chronological train/val/test, purging rows whose label window crosses a boundary."""
    ts = df["timestamp"]
    c1, c2 = ts.quantile(val_q), ts.quantile(test_q)
    purge = timedelta(days=lead_max_days)
    train = df.filter(pl.col("timestamp") < c1 - purge)
    val = df.filter((pl.col("timestamp") >= c1) & (pl.col("timestamp") < c2 - purge))
    test = df.filter(pl.col("timestamp") >= c2)
    return train, val, test, c2


# --------------------------------------------------------------------------- model + metrics

def fit(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    pos = max(int(y_train.sum()), 1)
    model = xgb.XGBClassifier(scale_pos_weight=(len(y_train) - pos) / pos, **XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def best_f1_threshold(y, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else 0.5


def row_metrics(y, scores, threshold) -> dict:
    pred = scores >= threshold
    tp = int((pred & (y == 1)).sum())
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 4) if y.any() else None,
        "roc_auc": round(float(roc_auc_score(y, scores)), 4) if 0 < y.sum() < len(y) else None,
        "threshold": round(threshold, 4),
        "precision": round(tp / max(int(pred.sum()), 1), 4),
        "recall": round(tp / max(int(y.sum()), 1), 4),
    }


def event_metrics(test: pl.DataFrame, scores: np.ndarray, threshold: float,
                  episodes: pl.DataFrame, test_start, cfg: LabelConfig) -> dict:
    """Per-episode recall/lead time and false-alarm days, as an operator would see them."""
    scored = test.select(KEY, "timestamp", "target").with_columns(
        pl.Series("alert", scores >= threshold))

    lead_max, lead_min = timedelta(days=cfg.lead_max_days), timedelta(days=cfg.lead_min_days)
    eligible = episodes.filter(pl.col("start") - lead_max >= test_start)
    windows = (
        eligible.select(KEY, "start")
        .join(scored.filter(pl.col("alert")), on=KEY)
        .filter(pl.col("timestamp").is_between(pl.col("start") - lead_max, pl.col("start") - lead_min))
        .group_by(KEY, "start")
        .agg(pl.col("timestamp").min().alias("first_alert"))
        .with_columns(((pl.col("start") - pl.col("first_alert")).dt.total_hours() / 24).alias("lead_days"))
    )

    days = scored.with_columns(pl.col("timestamp").dt.date().alias("day")).group_by(KEY, "day").agg(
        pl.col("alert").any(), (pl.col("alert") & (pl.col("target") == 1)).any().alias("true_alert"))
    false_days = days.filter(pl.col("alert") & ~pl.col("true_alert")).height
    inverter_months = days.height / 30

    n_events = eligible.height
    return {
        "episodes": n_events,
        "episodes_caught": windows.height,
        "event_recall": round(windows.height / n_events, 4) if n_events else None,
        "median_lead_days": round(float(windows["lead_days"].median()), 2) if windows.height else None,
        "false_alarm_days_per_inverter_month": round(false_days / max(inverter_months, 1e-9), 3),
    }


# --------------------------------------------------------------------------- experiments

def run_experiment(name, train, val, test, features, target, one_hot_plant=False):
    X_tr, y_tr = to_xy(train, features, target, one_hot_plant)
    X_va, y_va = to_xy(val, features, target, one_hot_plant)
    X_te, y_te = to_xy(test, features, target, one_hot_plant)
    model = fit(X_tr, y_tr, X_va, y_va)
    threshold = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
    scores = model.predict_proba(X_te)[:, 1]
    result = {"name": name, "features": len(X_tr.columns), "best_iteration": int(model.best_iteration),
              **row_metrics(y_te, scores, threshold)}
    print(f"  {name}: PR-AUC {result['pr_auc']} (prevalence {result['prevalence']}), ROC-AUC {result['roc_auc']}")
    return result, model, scores, threshold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=BACKEND_DIR / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "reports")
    parser.add_argument("--rebuild", action="store_true", help="ignore cached parquet files")
    parser.add_argument("--lead-min-days", type=float, default=LabelConfig.lead_min_days)
    parser.add_argument("--lead-max-days", type=float, default=LabelConfig.lead_max_days)
    parser.add_argument("--min-outage-hours", type=int, default=LabelConfig.min_hours)
    parser.add_argument("--export-model", choices=["original_leakfree", "full"],
                        help="save that experiment's model to models/ (xgboost JSON + feature list)")
    args = parser.parse_args(argv)

    cfg = LabelConfig(lead_min_days=args.lead_min_days, lead_max_days=args.lead_max_days,
                      min_hours=args.min_outage_hours)

    raw = load_dataset(args.raw_dir, args.cache_dir, rebuild=args.rebuild)
    df, episodes = prepare(raw, cfg)
    feats = feature_sets(df)

    legacy_fail = (pl.col("inv_power").abs() < 0.1) & pl.col("hour").is_between(6, 18)
    legacy_hours = df.filter(legacy_fail)
    summary = {
        "plants": sorted(df["plant_id"].unique().to_list()),
        "inverters": df[KEY].n_unique(),
        "inverter_hours": df.height,
        "period": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "legacy_failure_hours": legacy_hours.height,
        "legacy_failure_hours_at_dawn_dusk": legacy_hours.filter(pl.col("hour").is_in([6, 7, 17, 18])).height,
        "legacy_target_rate": round(float(df["legacy_target"].mean()), 4),
        "outage_hours": int(df["is_outage_hour"].sum()),
        "outage_episodes": episodes.height,
        "target_rate": round(float(df.filter(pl.col("label_ok"))["target"].mean()), 4),
        "label_config": cfg.__dict__,
    }
    print(json.dumps(summary, indent=2, default=str))

    results = {}
    models = {}

    # A: the hackathon methodology, reproduced on the rebuilt data
    print("A/B: legacy label")
    idx = np.arange(df.height)
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=42, stratify=df["legacy_target"].to_numpy())
    va, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=df["legacy_target"].to_numpy()[tmp])
    results["A_legacy_random_split"], *_ = run_experiment(
        "A legacy label, random split", df[tr], df[va], df[te], feats["original"], "legacy_target", one_hot_plant=True)

    train, val, test, test_start = time_split(df, cfg.lead_max_days)
    results["B_legacy_time_split"], *_ = run_experiment(
        "B legacy label, time split", train, val, test, feats["original"], "legacy_target", one_hot_plant=True)

    # C/D: new label, time split, only rows with a well-defined label
    print("C/D: outage-episode label, time split")
    labelled = df.filter(pl.col("label_ok"))
    train, val, test, test_start = time_split(labelled, cfg.lead_max_days)
    for key, fs, name in [("C_original_leakfree", "original_leakfree", "C original features minus leaky, time split"),
                          ("D_full", "full", "D all features + outage history, time split")]:
        res, model, scores, thr = run_experiment(name, train, val, test, feats[fs], "target")
        res["events"] = event_metrics(test, scores, thr, episodes, test_start, cfg)
        results[key] = res
        models[fs] = (model, feats[fs])

    # Sensitivity: does a looser outage definition (more, shorter episodes) change the picture?
    for min_hours in (1, 2):
        if min_hours == cfg.min_hours:
            continue
        alt_cfg = LabelConfig(lead_min_days=cfg.lead_min_days, lead_max_days=cfg.lead_max_days, min_hours=min_hours)
        alt_df, alt_episodes = prepare(raw, alt_cfg)
        alt = alt_df.filter(pl.col("label_ok"))
        a_tr, a_va, a_te, a_start = time_split(alt, cfg.lead_max_days)
        res, _, scores, thr = run_experiment(f"D full features, outages >= {min_hours}h ({alt_episodes.height} episodes)",
                                             a_tr, a_va, a_te, feats["full"], "target")
        res["events"] = event_metrics(a_te, scores, thr, alt_episodes, a_start, alt_cfg)
        results[f"S_min_outage_hours_{min_hours}"] = res

    # Rule baseline: "inverters that had outages recently will have more"
    base_val = val[BASELINE_SCORE].to_numpy().astype(float)
    base_test = test[BASELINE_SCORE].to_numpy().astype(float)
    thr = best_f1_threshold(val["target"].to_numpy(), base_val)
    results["baseline_recent_outages"] = {
        "name": f"baseline: {BASELINE_SCORE}",
        **row_metrics(test["target"].to_numpy(), base_test, thr),
        "events": event_metrics(test, base_test, thr, episodes, test_start, cfg),
    }

    # E: generalisation to an unseen plant
    plants = summary["plants"]
    if len(plants) > 1:
        print("E: leave-one-plant-out")
        for held_out in plants:
            source = labelled.filter(pl.col("plant_id") != held_out)
            cut = source["timestamp"].quantile(0.8)
            tr_p = source.filter(pl.col("timestamp") < cut - timedelta(days=cfg.lead_max_days))
            va_p = source.filter(pl.col("timestamp") >= cut)
            te_p = labelled.filter(pl.col("plant_id") == held_out)
            results[f"E_test_on_{held_out}"], *_ = run_experiment(
                f"E full features, test on {held_out}", tr_p, va_p, te_p, feats["full"], "target")

    if args.export_model:
        model, features = models[args.export_model]
        model_dir = BACKEND_DIR / "models"
        model.save_model(model_dir / "xgboost_solar_model.json")
        (model_dir / "feature_columns.json").write_text(json.dumps(features))
        print(f"Exported {args.export_model} model to {model_dir}")

    if hasattr(models.get("full", (None,))[0], "feature_importances_"):
        model, features = models["full"]
        top = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])[:10]
        summary["top_features_full"] = [[f, round(float(v), 4)] for f, v in top]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "results": results}
    (args.out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    (args.out_dir / "evaluation.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'evaluation.md'}")


def na(value):
    return "n/a" if value is None else value


def render_markdown(report: dict) -> str:
    s, r = report["summary"], report["results"]
    lines = [
        "# Model Evaluation",
        "",
        "Generated by `python -m src.ml.evaluate` (see `backend/src/ml/evaluate.py`).",
        "",
        "## Data",
        "",
        f"- Plants: {', '.join(s['plants'])} — {s['inverters']} inverters, {s['inverter_hours']:,} inverter-hours",
        f"- Period: {s['period'][0][:10]} to {s['period'][1][:10]}",
        f"- Legacy label: {s['legacy_failure_hours']:,} \"failure\" hours, "
        f"{s['legacy_failure_hours_at_dawn_dusk']:,} of them at 06-07h or 17-18h; "
        f"{s['legacy_target_rate']:.1%} of rows positive",
        f"- New label: {s['outage_hours']:,} outage hours in {s['outage_episodes']} episodes; "
        f"{s['target_rate']:.1%} of labelled rows positive "
        f"(episode starts in {s['label_config']['lead_min_days']:g}-{s['label_config']['lead_max_days']:g} days)",
        "",
        "## Row-level results (test set)",
        "",
        "PR-AUC of a random model equals the prevalence, so compare each PR-AUC with its own prevalence.",
        "",
        "| Experiment | Features | Test positives | Prevalence | PR-AUC | ROC-AUC | Precision | Recall |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for res in r.values():
        lines.append(f"| {res['name']} | {res.get('features', 1)} | {res['positives']:,} | {res['prevalence']} | "
                     f"{na(res['pr_auc'])} | {na(res['roc_auc'])} | {res['precision']} | {res['recall']} |")
    lines += [
        "",
        "## Event-level results (test set)",
        "",
        "An episode counts as caught if the model alerts at least once in the "
        f"{s['label_config']['lead_min_days']:g}-{s['label_config']['lead_max_days']:g} days before it starts. "
        "Thresholds were picked on the validation set (best F1). A model that alerts on most days "
        "catches every episode trivially, so read recall together with the false-alarm column "
        "(about 30 means alerting every day).",
        "",
        "| Model | Episodes | Caught | Event recall | Median lead (days) | False-alarm days / inverter-month |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for res in r.values():
        if "events" in res:
            e = res["events"]
            lines.append(f"| {res['name']} | {e['episodes']} | {e['episodes_caught']} | {na(e['event_recall'])} | "
                         f"{na(e['median_lead_days'])} | {e['false_alarm_days_per_inverter_month']} |")
    if "top_features_full" in s:
        lines += ["", "## Top features (full model, gain share)", ""]
        lines += [f"- `{f}`: {v}" for f, v in s["top_features_full"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
