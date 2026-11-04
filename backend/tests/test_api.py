"""End-to-end API tests on a synthetic logger export (uses the exported model in backend/models)."""
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.ml.serving import load_detector

pytestmark = pytest.mark.skipif(load_detector() is None, reason="no exported detector in backend/models")

START = datetime(2025, 3, 1, tzinfo=timezone.utc)


def logger_export(days=10, inverters=4, faulty=2, fault_from_day=8, fault_scale=0.4, seed=0) -> bytes:
    """5-minute raw export: inverters[i].power/temp following a shared sun curve with cloud noise."""
    rng = np.random.default_rng(seed)
    header = ["timestamp"] + [f"inverters[{i}].{f}" for i in range(inverters) for f in ("power", "temp")]
    lines = [",".join(header)]
    for step in range(days * 24 * 12):
        ts = START + timedelta(minutes=5 * step)
        hour = ts.hour + ts.minute / 60
        sun = max(0.0, np.sin(np.pi * (hour - 6) / 12)) * rng.uniform(0.8, 1.0)
        row = [str(int(ts.timestamp() * 1000))]
        for i in range(inverters):
            power = 300 * sun * rng.normal(1.0, 0.02)
            if i == faulty and step >= fault_from_day * 24 * 12:
                power *= fault_scale
            row += [f"{max(power, 0):.2f}", f"{25 + 30 * sun:.1f}"]
        lines.append(",".join(row))
    return "\n".join(lines).encode()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def detection(client):
    res = client.post("/detect", files=[("files", ("PLANT-A.raws.csv", io.BytesIO(logger_export()), "text/csv"))])
    assert res.status_code == 200, res.text
    return res.json()


def test_health_reports_detector(client):
    body = client.get("/health").json()
    assert body["detector_loaded"] is True
    assert body["status"] == "healthy"


def test_degraded_inverter_is_the_only_alarm(detection):
    by_id = {r["inverter_id"]: r for r in detection["results"]}
    assert by_id["PLANT-A#2"]["status"] == "ALARM"
    assert by_id["PLANT-A#2"]["energy_ratio_latest_day"] < 0.6
    assert all(r["status"] != "ALARM" for k, r in by_id.items() if k != "PLANT-A#2")
    assert detection["results"][0]["inverter_id"] == "PLANT-A#2"  # sorted worst first
    assert detection["summary"]["inverters"] == 4


def test_results_carry_chart_series_and_evidence(detection):
    worst = detection["results"][0]
    assert worst["hourly"] and {"actual_pu", "expected_pu", "cusum", "alarm"} <= set(worst["hourly"][0])
    signals = {e["signal"] for e in worst["evidence"]}
    assert {"energy_vs_expected", "zero_output_hours", "output_vs_peers"} <= signals


def test_rule_based_report_without_llm(client, detection, monkeypatch):
    monkeypatch.setattr("src.genai.narrative.is_llm_available", lambda: False)
    worst = detection["results"][0]
    inverter = {k: worst[k] for k in ("inverter_id", "status", "risk_score", "energy_ratio_latest_day",
                                      "energy_ratio_window", "alarm_hours", "first_alarm", "evidence")}
    body = client.post("/explain", json={"inverter": inverter}).json()
    assert body["source"] == "rules"
    assert body["report"]["likelyCauses"]
    assert body["report"]["status"] == "ALARM"


def test_query_fallback_lists_flagged_inverters(client, detection, monkeypatch):
    monkeypatch.setattr("src.genai.assistant.is_llm_available", lambda: False)
    fleet = [{k: r[k] for k in ("inverter_id", "status", "risk_score", "alarm_hours")} for r in detection["results"]]
    answer = client.post("/query", json={"question": "Which inverters are in alarm?", "fleet": fleet}).json()["answer"]
    assert "PLANT-A#2" in answer


def test_rejects_too_few_inverters(client):
    data = logger_export(inverters=2, faulty=-1)
    res = client.post("/detect", files=[("files", ("TWO.raws.csv", io.BytesIO(data), "text/csv"))])
    assert res.status_code == 400
    assert "at least 3 inverters" in res.json()["detail"]


def test_rejects_non_logger_csv(client):
    res = client.post("/detect", files=[("files", ("x.csv", io.BytesIO(b"timestamp,a\n1,2\n"), "text/csv"))])
    assert res.status_code == 400


@pytest.mark.parametrize("origin", ["https://solar-sight-six.vercel.app", "http://localhost:3000"])
def test_cors_preflight_allows_deployed_frontend(client, origin):
    res = client.options("/detect", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == origin
