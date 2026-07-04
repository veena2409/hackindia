"""
main.py
-------
FastAPI server for the MedGemma Brain MRI Chat app.

Endpoints
---------
GET  /health          -> {"status": "ok"}
GET  /disclaimer      -> {"disclaimer": "..."}
POST /new_session     -> {"session_id": "<uuid>"}
POST /chat            -> SSE stream of text chunks
POST /analyze         -> JSON quality assessment + heatmap_base64

Run:
    uvicorn main:app --reload --port 8000 --app-dir .
"""

from __future__ import annotations

import base64
import io
import json
import logging
import uuid
import datetime
from typing import Optional, List

from pathlib import Path

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, Form, UploadFile
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles

# Import the isolated MedGemma client (Phase 1)
from medgemma_client import ask_medgemma_stream, analyze_scan

# Paths (resolve relative to this file so the server can be started from anywhere)
_BACKEND_DIR  = Path(__file__).parent
_PROJECT_DIR  = _BACKEND_DIR.parent
_DIST_DIR     = _PROJECT_DIR / "frontend" / "dist"
_SAMPLES_DIR  = _PROJECT_DIR / "frontend" / "public" / "samples"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mri_chat_server")

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MedGemma Brain MRI Chat",
    description=(
        "Educational image-quality assistant powered by MedGemma. "
        "NOT a clinical device."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    # Allow all origins — safe for this educational tool; tighten if needed.
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Medical disclaimer (single source of truth)
# ---------------------------------------------------------------------------
_DISCLAIMER = (
    "This tool is an educational helper for understanding MRI image quality. "
    "It is NOT a medical device and does NOT provide medical diagnoses or clinical advice. "
    "All outputs are for informational and learning purposes only. "
    "Always consult a qualified medical professional before making any health-related decisions. "
    "Do not use this tool for clinical, diagnostic, or treatment purposes."
)

# ---------------------------------------------------------------------------
# In-memory session store
# { session_id: {"history": [...], "image_bytes": bytes | None} }
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}
_MAX_HISTORY_TURNS = 20

# In-memory analysis history list
# Each entry: {scan_id, artifact_type, severity, quality_score, timestamp}
_analysis_history: List[dict] = []


def _get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {"history": [], "image_bytes": None}
    return _sessions[session_id]


def _trim_history(history: list) -> list:
    max_entries = _MAX_HISTORY_TURNS * 2
    return history[-max_entries:] if len(history) > max_entries else history


# ---------------------------------------------------------------------------
# Sharpness heatmap (OpenCV + numpy — no AI)
# ---------------------------------------------------------------------------

def make_sharpness_map(image_bytes: bytes) -> str:
    """
    Compute a local sharpness map using variance-of-Laplacian over a sliding
    window. Returns the result as a base64-encoded PNG of the heatmap blended
    over the original image.

    Never raises — returns empty string on any failure.
    """
    try:
        import numpy as np
        import cv2

        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Variance of Laplacian over sliding window (patch-level sharpness)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        # Blur the squared Laplacian to get local variance map
        lap_sq = (lap ** 2).astype(np.float32)
        kernel_size = max(31, (min(h, w) // 16) | 1)  # odd, at least 31px
        local_var = cv2.GaussianBlur(lap_sq, (kernel_size, kernel_size), 0)

        # Normalize to 0-255
        mn, mx = local_var.min(), local_var.max()
        if mx > mn:
            sharpness_norm = ((local_var - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            sharpness_norm = np.zeros_like(local_var, dtype=np.uint8)

        # Apply JET colormap (blue=blurry, red=sharp)
        heatmap = cv2.applyColorMap(sharpness_norm, cv2.COLORMAP_JET)

        # Alpha-blend heatmap over original image
        alpha = 0.55
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

        # Encode to PNG and return as base64
        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf.tobytes()).decode("ascii")

    except Exception as exc:
        logger.error("make_sharpness_map failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Startup: validate MedGemma config and warn immediately if token is missing
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup_config_check():
    """
    Runs once when uvicorn starts. Logs a clear WARNING if the HuggingFace
    token has not been set, so the problem is obvious before the first request.
    """
    import os
    token  = os.environ.get("MEDGEMMA_AUTH_TOKEN", "").strip()
    model  = os.environ.get("MEDGEMMA_MODEL",      "google/medgemma-1.5-4b-it").strip()
    style  = os.environ.get("MEDGEMMA_API_STYLE",  "huggingface").strip()

    # Also check cached HF token (set by `login()` or `huggingface-cli login`)
    cached_token = ""
    if style == "huggingface" and (not token or "YOUR_TOKEN" in token):
        try:
            import sys
            sys.path.insert(0, str(_BACKEND_DIR / ".deps"))
            from huggingface_hub import get_token  # type: ignore
            cached_token = get_token() or ""
        except Exception:
            pass

    placeholder = not token or "YOUR_TOKEN" in token
    resolved    = not placeholder or bool(cached_token)

    if resolved:
        effective_tok = cached_token if placeholder else token
        logger.info(
            "MedGemma config OK — style=%s  model=%s  token=%s",
            style, model, effective_tok[:8] + "..." + effective_tok[-4:],
        )
    else:
        logger.warning(
            "="*70
        )
        logger.warning("  API Key NOT SET — AI replies will use the fallback message.")
        logger.warning("  To fix, paste your GEMINI_API_KEY into backend/.env")
        logger.warning("  Get your free key at: https://aistudio.google.com/app/apikey")
        logger.warning(
            "="*70
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get("/config_status", tags=["meta"])
async def config_status():
    """
    Returns the current MedGemma configuration state.
    Useful to confirm the token and style are loaded correctly.
    Visit: http://localhost:8000/config_status
    """
    import os
    token  = os.environ.get("MEDGEMMA_AUTH_TOKEN", "").strip()
    model  = os.environ.get("MEDGEMMA_MODEL",     "google/medgemma-1.5-4b-it").strip()
    style  = os.environ.get("MEDGEMMA_API_STYLE", "huggingface").strip()

    placeholder   = not token or "YOUR_TOKEN" in token
    cached_token  = ""
    if placeholder:
        try:
            from huggingface_hub import get_token  # type: ignore
            cached_token = get_token() or ""
        except Exception:
            pass

    effective    = cached_token if (placeholder and cached_token) else token
    token_ok     = bool(effective and "YOUR_TOKEN" not in effective)
    token_source = "env" if (not placeholder) else ("cache" if cached_token else "missing")

    return {
        "api_style":    style,
        "model":        model,
        "token_status": "ok"         if token_ok else "missing",
        "token_source": token_source,
        "token_prefix": effective[:8] + "..." if (token_ok and len(effective) > 8) else "(not set)",
        "ready":        token_ok,
        "next_step":    None if token_ok else (
            "Set GEMINI_API_KEY=... in backend/.env "
            "Get your free key at: https://aistudio.google.com/app/apikey"
        ),
    }




@app.get("/disclaimer", tags=["meta"])
async def disclaimer():
    return {"disclaimer": _DISCLAIMER}


@app.post("/new_session", tags=["session"])
async def new_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = {"history": [], "image_bytes": None}
    logger.info("New session created: %s", sid)
    return {"session_id": sid}


@app.post("/chat", tags=["chat"])
async def chat(
    session_id: str = Form(...),
    message: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
):
    """
    Stream a MedGemma reply for the given user message.
    """
    if not session_id or not session_id.strip():
        def _err():
            yield "No session ID provided. Please call /new_session first."
        return StreamingResponse(_err(), media_type="text/plain")

    session = _get_or_create_session(session_id)

    # Handle image upload
    image_bytes: Optional[bytes] = None
    if image is not None:
        try:
            image_bytes = await image.read()
            if len(image_bytes) == 0:
                image_bytes = None
            else:
                session["image_bytes"] = image_bytes
                logger.info(
                    "[%s] New image uploaded (%d bytes, type=%s)",
                    session_id, len(image_bytes), image.content_type,
                )
        except Exception as exc:
            logger.error("[%s] Failed to read uploaded image: %s", session_id, exc)
            image_bytes = None
    else:
        image_bytes = session.get("image_bytes")
        if image_bytes:
            logger.info(
                "[%s] Reusing session image (%d bytes) for context.",
                session_id, len(image_bytes),
            )

    history_snapshot = list(session["history"])

    logger.info(
        "[%s] /chat -- message=%r  image=%s  history_len=%d",
        session_id, message[:80], image_bytes is not None, len(history_snapshot),
    )

    def generate():
        collected_chunks: list[str] = []
        try:
            for chunk in ask_medgemma_stream(
                user_message=message,
                image_bytes=image_bytes,
                history=history_snapshot,
            ):
                collected_chunks.append(chunk)
                yield chunk
        except Exception as exc:
            logger.exception("[%s] Unexpected error during streaming: %s", session_id, exc)
            fallback = (
                "I'm sorry -- something went wrong on my end. "
                "Please try again in a moment."
            )
            yield fallback
            collected_chunks.append(fallback)
            return

        full_reply = "".join(collected_chunks)
        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "assistant", "content": full_reply})
        session["history"] = _trim_history(session["history"])

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/analyze", tags=["analyze"])
async def analyze(image: UploadFile = File(...)):
    """
    Run a structured quality analysis on the uploaded brain MRI.

    Returns JSON with:
        artifact_present, artifact_type, severity, quality_score,
        region, recommendation, explanation, heatmap_base64,
        scan_id, timestamp
    """
    try:
        image_bytes = await image.read()
        if not image_bytes:
            return JSONResponse({"error": "Empty image"}, status_code=400)
    except Exception as exc:
        logger.error("/analyze: failed to read image: %s", exc)
        return JSONResponse({"error": "Could not read image"}, status_code=400)

    logger.info("/analyze: received %d bytes (type=%s)", len(image_bytes), image.content_type)

    # Run MedGemma analysis and sharpness map in sequence
    # (analyze_scan and make_sharpness_map both never raise)
    result = analyze_scan(image_bytes)
    heatmap_b64 = make_sharpness_map(image_bytes)

    # Attach heatmap and metadata
    scan_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    result["heatmap_base64"] = heatmap_b64
    result["scan_id"] = scan_id
    result["timestamp"] = timestamp

    # Store summary in history
    _analysis_history.append({
        "scan_id": scan_id,
        "artifact_type": result.get("artifact_type", "none"),
        "severity": result.get("severity", "none"),
        "quality_score": result.get("quality_score"),
        "timestamp": timestamp,
    })

    logger.info(
        "/analyze: scan_id=%s  artifact=%s  severity=%s  score=%s",
        scan_id, result.get("artifact_type"), result.get("severity"), result.get("quality_score"),
    )

    return JSONResponse(result)



@app.get("/analysis_history", tags=["analyze"])
async def analysis_history():
    """Return the in-memory analysis history (most recent first)."""
    return {"history": list(reversed(_analysis_history))}


# ---------------------------------------------------------------------------
# Static file serving — mounted AFTER all API routes.
# Serves the built Vite dist/ and sample images on the same port as the API,
# so a single cloudflared/ngrok tunnel exposes the whole app.
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).parent
_PROJECT_DIR = _BACKEND_DIR.parent
_DIST_DIR    = _PROJECT_DIR / "frontend" / "dist"
_SAMPLES_DIR = _PROJECT_DIR / "frontend" / "public" / "samples"


# Serve sample images at /samples/*
if _SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(_SAMPLES_DIR)), name="samples")
    logger.info("Serving /samples from %s", _SAMPLES_DIR)


# Serve built Vite assets and SPA fallback
if _DIST_DIR.exists():
    _assets_dir = _DIST_DIR / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    # Explicit root
    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(str(_DIST_DIR / "index.html"))

    # SPA fallback — must be last
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = _DIST_DIR / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_DIST_DIR / "index.html"))

    logger.info("Serving built frontend SPA from %s", _DIST_DIR)
else:
    logger.warning(
        "No production build at %s — run 'npm run build' in frontend/.", _DIST_DIR
    )

