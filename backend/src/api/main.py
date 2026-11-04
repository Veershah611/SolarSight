from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.routes import router
from src.ml.serving import load_detector
import logging
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading underperformance detector...")
    app.state.detector = load_detector()
    if app.state.detector is None:
        logging.warning("No detector in backend/models; /detect will return 503.")
    yield
    app.state.detector = None

app = FastAPI(title="SolarSight API", description="Same-day detection of underperforming solar inverters", lifespan=lifespan)

# CORS — comma-separated CORS_ORIGINS overrides the defaults (local Vite/Docker + Vercel)
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",  # Vite dev server and Docker frontend
    "http://localhost:5173",  # Vite default port
    "https://latent-space-hackamined.vercel.app",
    "https://solar-sight-six.vercel.app",
]
# Browsers send origins without a trailing slash, so strip one if it was pasted into the env var
cors_origins = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()] or DEFAULT_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
