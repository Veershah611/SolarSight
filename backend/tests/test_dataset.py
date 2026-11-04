from datetime import datetime, timezone

import polars as pl

from src.ml.dataset import load_hourly


def write_export(path, rows):
    """Minimal raw logger export: epoch-ms timestamp and two inverters' power/temp."""
    header = "timestamp,inverters[0].power,inverters[0].temp,inverters[1].power,inverters[1].temp,smu[0].string1,smu[0].string2"
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines))


def ms(hour, minute=0):
    return int(datetime(2025, 1, 1, hour, minute, tzinfo=timezone.utc).timestamp() * 1000)


def test_nan_readings_do_not_poison_hourly_means(tmp_path):
    path = tmp_path / "ICR2-TEST.raws.csv"
    write_export(path, [
        (ms(10, 0), 100, 40, 50, 30, 5, 6),
        (ms(10, 5), "NaN", 41, 50, 30, 5, 6),
        (ms(10, 10), 110, "nan", 50, 30, "NaN", 6),
    ])
    df = load_hourly([path], plant="p", log=lambda *_: None)
    a = df.filter(pl.col("inverter_id") == "ICR2-TEST#0").row(0, named=True)
    assert a["inv_power"] == 105
    assert a["inv_temp"] == 40.5
    assert a["smu_string1"] == 5


def test_inverters_from_two_files_keep_separate_ids(tmp_path):
    rows = [(ms(10), 100, 40, 50, 30, 5, 6)]
    write_export(tmp_path / "LOGGER-A.raws.csv", rows)
    write_export(tmp_path / "LOGGER-B.raws.csv", rows)
    df = load_hourly([tmp_path / "LOGGER-A.raws.csv", tmp_path / "LOGGER-B.raws.csv"], plant="p", log=lambda *_: None)
    assert sorted(df["inverter_id"]) == ["LOGGER-A#0", "LOGGER-A#1", "LOGGER-B#0", "LOGGER-B#1"]
    assert df["inverter_idx"].n_unique() == 4
