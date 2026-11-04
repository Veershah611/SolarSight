"""Operator Q&A grounded in the latest detection results."""
import json
import logging
from typing import List

from src.api.schemas import InverterContext
from src.genai.llm_client import generate_text, is_llm_available


def _fleet_table(fleet: List[InverterContext]) -> str:
    rows = ["inverter | status | shortfall % (latest day) | energy ratio (latest day) | alarm hours"]
    rows += [f"{f.inverter_id} | {f.status} | {f.risk_score} | {f.energy_ratio_latest_day} | {f.alarm_hours}"
             for f in fleet]
    return "\n".join(rows)


def _fallback(question: str, fleet: List[InverterContext]) -> str:
    flagged = [f for f in fleet if f.status != "OK"]
    if not fleet:
        return "No detection results yet. Upload a data-logger export in the Analyzer tab first."
    lines = [f"{len(flagged)} of {len(fleet)} inverters need attention:" if flagged
             else f"All {len(fleet)} inverters are performing in line with their peers."]
    lines += [f"- {f.inverter_id}: {f.status}, {f.risk_score:.0f}% below expected on the latest day, "
              f"{f.alarm_hours} alarm hours" for f in flagged]
    lines.append("\n(Set GEMINI_API_KEY in backend/.env for free-form answers.)")
    return "\n".join(lines)


def answer_query(question: str, inverter: InverterContext | None, fleet: List[InverterContext],
                 parameters: str | None = None) -> str:
    if not is_llm_available():
        return _fallback(question, fleet)

    selected = json.dumps(inverter.model_dump(), indent=2) if inverter else "None selected"
    prompt = (
        "You are an assistant for a solar plant operator. The data below comes from a detector that compares "
        "each inverter's output with what its plant peers imply it should produce (energy ratio 1.0 = as "
        "expected); ALARM means a sustained shortfall, WATCH a smaller or earlier one.\n\n"
        f"Fleet:\n{_fleet_table(fleet) if fleet else 'No detection results uploaded yet.'}\n\n"
        f"Selected inverter detail:\n{selected}\n\n"
        f"Operator notes: {parameters or 'none'}\n\n"
        "Rules:\n"
        "- Answer from the data above; if it doesn't contain the answer, say so instead of guessing numbers.\n"
        "- For maintenance questions, give concrete field actions that match the evidence.\n"
        "- If the question is unrelated to solar operations, say you can't help with it.\n"
        "- Be concise.\n\n"
        f"Question: {question}"
    )
    try:
        return generate_text(prompt) or _fallback(question, fleet)
    except Exception as e:
        logging.error(f"Gemini query failed: {e}")
        return f"The AI assistant is unavailable right now ({e}).\n\n" + _fallback(question, fleet)
