"""Maintenance report for one inverter: Gemini when configured, otherwise rule-based.

Both paths return the same JSON shape, so the dashboard renders either:
    {status, headline, summary, estimatedLossPct,
     likelyCauses: [{rank, name, category, confidence, evidence, action}], checks: [str]}
"""
import json
import logging

from src.api.schemas import InverterContext
from src.genai.llm_client import generate_json, is_llm_available

CATEGORIES = ["electrical", "string", "thermal", "soiling_shading", "communication", "other"]


def _evidence_map(inv: InverterContext) -> dict:
    return {e.signal: e for e in inv.evidence}


def _value(ev: dict, signal: str):
    return ev[signal].value if signal in ev else None


def rule_based_report(inv: InverterContext) -> dict:
    """Deterministic diagnosis from the detector's evidence signals."""
    ev = _evidence_map(inv)
    ratio = _value(ev, "energy_vs_expected")
    zero_hours = _value(ev, "zero_output_hours") or 0
    causes = []

    def add(name, category, confidence, evidence, action):
        causes.append({"rank": len(causes) + 1, "name": name, "category": category,
                       "confidence": confidence, "evidence": evidence, "action": action})

    if zero_hours >= 3 or (ratio is not None and ratio < 0.2):
        add("Inverter trip or AC-side outage", "electrical", 80,
            f"{int(zero_hours)} daylight hours with no output while peers were producing",
            "Read the inverter fault log, check the AC breaker and grid connection, then restart the inverter.")
    if "string_balance" in ev:
        now, typical = ev["string_balance"].value, ev["string_balance"].typical
        if now is not None and typical is not None and now < typical - 0.15:
            add("String or combiner-box fault", "string", 70,
                f"Weakest/strongest string current fell to {now:.2f} (typically {typical:.2f})",
                "Measure string currents and check fuses and connectors in the combiner box.")
    if "temp_vs_peers_c" in ev:
        now, typical = ev["temp_vs_peers_c"].value, ev["temp_vs_peers_c"].typical
        if now is not None and typical is not None and now > typical + 5 and (ratio is None or ratio < 0.95):
            add("Thermal derating", "thermal", 60,
                f"Running {now - typical:.1f} °C hotter relative to peers than usual",
                "Inspect cooling fans, air filters and enclosure ventilation.")
    if inv.status != "OK" and ratio is not None and 0.2 <= ratio < 0.95 and not causes:
        add("Partial output loss (soiling, shading or a failed MPPT input)", "soiling_shading", 45,
            f"Produced {ratio:.0%} of expected output without dropping to zero",
            "Inspect the array for soiling or new shading and compare the inverter's MPPT input currents.")
    if inv.status != "OK" and not causes:
        add("Intermittent underperformance", "other", 30,
            f"{inv.alarm_hours} alarm hours in the uploaded window",
            "Review the inverter's event log around the first alarm and keep it under observation.")

    loss = round(max(0.0, 1 - ratio) * 100, 1) if ratio is not None else inv.risk_score
    if inv.status == "OK":
        headline = "Performing in line with peers"
        summary = (f"{inv.inverter_id} produced {ratio:.0%} of its expected output on the latest day, "
                   "within normal variation of its peers." if ratio is not None else
                   f"{inv.inverter_id} shows no sustained shortfall against its peers.")
    else:
        top = causes[0]["name"]
        summary = (f"{inv.inverter_id} produced {ratio:.0%} of its expected output on the latest day "
                   f"({inv.alarm_hours} alarm hours in the window). Most likely cause: {top}."
                   if ratio is not None else f"{inv.inverter_id} has {inv.alarm_hours} alarm hours. Most likely cause: {top}.")
        headline = causes[0]["name"]

    checks = [c["action"] for c in causes] or ["No action needed; keep monitoring."]
    return {"status": inv.status, "headline": headline, "summary": summary,
            "estimatedLossPct": loss, "likelyCauses": causes, "checks": checks}


def _prompt(inv: InverterContext, parameters: str | None) -> str:
    facts = {
        "inverter_id": inv.inverter_id,
        "status": inv.status,
        "energy_shortfall_latest_day_pct": inv.risk_score,
        "energy_ratio_latest_day": inv.energy_ratio_latest_day,
        "energy_ratio_whole_upload": inv.energy_ratio_window,
        "alarm_hours": inv.alarm_hours,
        "first_alarm": inv.first_alarm,
        "evidence": [e.model_dump() for e in inv.evidence],
    }
    return (
        "You are a solar O&M engineer. A detector compares each inverter's output with what its plant peers "
        "imply it should produce (energy_ratio 1.0 = as expected) and raises a CUSUM alarm on sustained "
        "shortfall. Diagnose this inverter using ONLY the facts below; do not invent measurements.\n\n"
        f"Facts:\n{json.dumps(facts, indent=2)}\n\n"
        f"Operator notes: {parameters or 'none'}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '{"status": "ALARM|WATCH|OK", "headline": "<= 8 words", '
        '"summary": "2-3 sentences citing the numbers", "estimatedLossPct": number, '
        '"likelyCauses": [{"rank": 1, "name": str, "category": one of '
        f"{CATEGORIES}, "
        '"confidence": 0-100, "evidence": "which fact supports it", "action": "what the technician should do"}], '
        '"checks": ["ordered on-site checks"]}\n'
        "List at most 3 causes. If status is OK, return an empty likelyCauses list."
    )


def generate_report(inv: InverterContext, parameters: str | None = None) -> tuple[str, dict]:
    """(source, report): Gemini's report if available and well-formed, else the rule-based one."""
    if is_llm_available():
        try:
            report = generate_json(_prompt(inv, parameters))
            if report and isinstance(report.get("likelyCauses"), list) and report.get("summary"):
                return "gemini", report
            logging.warning("Gemini report missing required keys; using rules")
        except Exception as e:
            logging.error(f"Gemini report failed: {e}")
    return "rules", rule_based_report(inv)
