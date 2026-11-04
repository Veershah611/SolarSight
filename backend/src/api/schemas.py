from typing import Any, List, Literal

from pydantic import BaseModel

Status = Literal["ALARM", "WATCH", "OK"]


class Evidence(BaseModel):
    signal: str
    value: float | None
    typical: float | None = None
    description: str = ""


class DailyPoint(BaseModel):
    day: str
    energy_ratio: float | None
    alarm: bool


class HourPoint(BaseModel):
    timestamp: str
    actual_pu: float | None
    expected_pu: float | None
    cusum: float | None
    alarm: bool


class InverterDetection(BaseModel):
    inverter_id: str
    status: Status
    risk_label: Literal["HIGH", "MEDIUM", "LOW"]
    risk_score: float  # energy shortfall vs expected on the latest day, %
    energy_ratio_latest_day: float | None
    energy_ratio_window: float | None
    alarm_hours: int
    first_alarm: str | None
    last_alarm: str | None
    judged_hours: int
    evidence: List[Evidence]
    daily: List[DailyPoint]
    hourly: List[HourPoint]


class DetectionSummary(BaseModel):
    inverters: int
    period_start: str
    period_end: str
    latest_day: str
    judged_hours: int
    inverters_in_alarm: int
    inverters_on_watch: int
    detector: dict[str, Any]


class DetectResponse(BaseModel):
    summary: DetectionSummary
    results: List[InverterDetection]


class InverterContext(BaseModel):
    """The parts of an InverterDetection the GenAI endpoints need (no chart series)."""
    inverter_id: str
    status: Status
    risk_score: float
    energy_ratio_latest_day: float | None = None
    energy_ratio_window: float | None = None
    alarm_hours: int = 0
    first_alarm: str | None = None
    evidence: List[Evidence] = []


class ExplainRequest(BaseModel):
    inverter: InverterContext
    parameters: str | None = None


class ExplainResponse(BaseModel):
    inverter_id: str
    source: Literal["gemini", "rules"]
    report: dict[str, Any]


class QueryRequest(BaseModel):
    question: str
    inverter: InverterContext | None = None
    fleet: List[InverterContext] = []
    parameters: str | None = None
