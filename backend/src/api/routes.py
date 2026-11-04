import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import DetectResponse, ExplainRequest, ExplainResponse, QueryRequest
from src.genai.assistant import answer_query
from src.genai.narrative import generate_report
from src.ml.dataset import load_hourly
from src.ml.serving import analyse

router = APIRouter()

MAX_UPLOAD_BYTES = 60 * 1024 * 1024


@router.get("/health")
async def health_check(fast_req: Request) -> Dict[str, Any]:
    detector = getattr(fast_req.app.state, "detector", None)
    return {
        "status": "healthy" if detector else "degraded",
        "detector_loaded": detector is not None,
        "detector": {"k": detector.cfg.k, "h": detector.cfg.h,
                     "trained_until": detector.meta.get("trained_until")} if detector else None,
    }


@router.post("/detect", response_model=DetectResponse)
async def detect(fast_req: Request, files: List[UploadFile] = File(...)):
    """Score one or more raw data-logger exports from the same plant."""
    detector = getattr(fast_req.app.state, "detector", None)
    if detector is None:
        raise HTTPException(503, "Detector model not found in backend/models; run evaluate_detection --export-model.")

    with tempfile.TemporaryDirectory() as tmp:
        paths, total = [], 0
        for i, upload in enumerate(files):
            content = await upload.read()
            total += len(content)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"Uploads are limited to {MAX_UPLOAD_BYTES // 2**20} MB in total.")
            name = Path(upload.filename or f"upload_{i}.csv").name
            # one folder per file keeps the original name (it prefixes inverter ids)
            path = Path(tmp) / str(i) / name
            path.parent.mkdir()
            path.write_bytes(content)
            paths.append(path)
        try:
            # CPU-bound: keep the event loop free for other requests
            return await run_in_threadpool(_detect, paths, detector)
        except ValueError as e:
            raise HTTPException(400, str(e))


def _detect(paths, detector):
    return analyse(load_hourly(paths, plant="upload", log=lambda *_: None), detector)


# Plain `def` so FastAPI runs the blocking LLM calls in its threadpool
@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    source, report = generate_report(request.inverter, request.parameters)
    return ExplainResponse(inverter_id=request.inverter.inverter_id, source=source, report=report)


@router.post("/query")
def query(request: QueryRequest):
    return {"answer": answer_query(request.question, request.inverter, request.fleet, request.parameters)}
