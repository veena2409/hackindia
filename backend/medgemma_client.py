"""
medgemma_client.py
------------------
A standalone, swappable module that talks to a MedGemma inference endpoint.

Public API
----------
ask_medgemma(user_message, image_bytes=None, history=None)        -> str
ask_medgemma_stream(user_message, image_bytes=None, history=None) -> Iterator[str]
analyze_scan(image_bytes)                                          -> dict

Supported API styles (set MEDGEMMA_API_STYLE in .env):
  gemini       - Google Gemini API via google-genai (default)
  huggingface  - HuggingFace InferenceClient (official HF SDK)
  openai       - any OpenAI-compatible HTTP endpoint (raw httpx)
  vertex       - Google Vertex AI online-prediction endpoint
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Generator, List, Optional

import httpx
from dotenv import load_dotenv

# Load .env from the same directory as this file
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("medgemma_client")

_SYSTEM_PROMPT = (
    "You are a friendly MRI image-quality assistant for a learning tool. "
    "Look at the uploaded brain MRI and describe, in plain everyday language, "
    "whether it looks clear or shows motion artifacts such as blurring, ghosting, "
    "or banding, and whether the image would be usable for review. "
    "Be calm, warm, and never alarming. "
    "You are NOT a doctor and must NOT give a diagnosis. "
    "Always remind the user, briefly, that a qualified professional should review "
    "the actual scan. "
    "Keep answers short and easy to understand unless asked for detail."
)

_FALLBACK_REPLY = (
    "I'm sorry - I wasn't able to reach the AI model right now. "
    "Please check your internet connection and try again in a moment. "
    "If the problem continues, contact the person who set up this tool."
)


# =============================================================================
# Config
# =============================================================================

def _read_config() -> dict:
    # Try multiple token variables to be friendly
    token    = (os.environ.get("GEMINI_API_KEY") or os.environ.get("MEDGEMMA_AUTH_TOKEN", "")).strip()
    model    = os.environ.get("MEDGEMMA_MODEL", "gemini-1.5-flash").strip()
    style    = os.environ.get("MEDGEMMA_API_STYLE", "gemini").strip().lower()
    endpoint = os.environ.get("MEDGEMMA_ENDPOINT_URL", "").strip()
    timeout  = float(os.environ.get("MEDGEMMA_TIMEOUT", "120"))

    # Auto-read HF token cached by `huggingface-cli login` if style is huggingface
    if not token and style == "huggingface":
        try:
            from huggingface_hub import get_token  # type: ignore
            cached = get_token()
            if cached:
                token = cached
                logger.info("Using cached HuggingFace token from huggingface_hub store.")
        except Exception:
            pass

    if not token:
        raise ValueError(
            "API Key not set.\n"
            "  For Gemini: Set GEMINI_API_KEY in backend/.env\n"
            "  For HuggingFace: Set MEDGEMMA_AUTH_TOKEN in backend/.env"
        )

    if style not in ("gemini", "huggingface", "openai", "vertex"):
        logger.warning("Unknown MEDGEMMA_API_STYLE %r - defaulting to gemini.", style)
        style = "gemini"

    if style in ("openai", "vertex") and not endpoint:
        raise ValueError(
            f"MEDGEMMA_API_STYLE={style} requires MEDGEMMA_ENDPOINT_URL in .env"
        )

    return {"token": token, "model": model, "style": style,
            "endpoint": endpoint, "timeout": timeout}


# =============================================================================
# Message builder (OpenAI Format)
# =============================================================================

def _image_to_data_url(image_bytes: bytes) -> str:
    mime = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
    b64  = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_messages(user_message: str,
                    image_bytes: Optional[bytes],
                    history: Optional[List[dict]]) -> list:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        for turn in history:
            role, content = turn.get("role", "user"), turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    if image_bytes:
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes)}},
            {"type": "text",      "text": user_message},
        ]})
    else:
        messages.append({"role": "user", "content": user_message})
    return messages


# =============================================================================
# Gemini API Backend (via google-genai)
# =============================================================================

def _gemini_contents(user_message: str,
                     image_bytes: Optional[bytes],
                     history: Optional[List[dict]]) -> tuple[list, str]:
    """Convert into Gemini native format. Returns (contents, system_instruction)"""
    contents = []
    
    # Process history
    if history:
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                # Gemini roles are "user" and "model"
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})
    
    # Process current turn
    current_parts = []
    if image_bytes:
        mime_type = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
        b64_data = base64.b64encode(image_bytes).decode("ascii")
        current_parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": b64_data
            }
        })
    
    current_parts.append({"text": user_message})
    contents.append({"role": "user", "parts": current_parts})
    
    return contents, _SYSTEM_PROMPT


def _gemini_stream(cfg: dict, user_message: str, image_bytes: Optional[bytes], history: Optional[List[dict]]) -> Generator[str, None, None]:
    contents, system_instruction = _gemini_contents(user_message, image_bytes, history)
    logger.info("-> Gemini streaming (raw HTTP) model=%s", cfg["model"])
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:streamGenerateContent?alt=sse&key={cfg['token']}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"temperature": 0.2}
    }
    
    with httpx.Client(timeout=cfg["timeout"]) as c:
        with c.stream("POST", url, json=payload, headers={"Content-Type": "application/json"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        if "candidates" in chunk and chunk["candidates"]:
                            parts = chunk["candidates"][0].get("content", {}).get("parts", [])
                            if parts:
                                yield parts[0].get("text", "")
                    except json.JSONDecodeError:
                        continue

def _gemini_complete(cfg: dict, user_message: str, image_bytes: Optional[bytes], history: Optional[List[dict]]) -> str:
    contents, system_instruction = _gemini_contents(user_message, image_bytes, history)
    logger.info("-> Gemini complete (raw HTTP) model=%s", cfg["model"])
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent?key={cfg['token']}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"temperature": 0.2}
    }
    
    with httpx.Client(timeout=cfg["timeout"]) as c:
        resp = c.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        if "candidates" in data and data["candidates"]:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return ""


# =============================================================================
# HuggingFace InferenceClient backend
# =============================================================================

def _hf_stream(cfg: dict, messages: list) -> Generator[str, None, None]:
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except ImportError:
        raise RuntimeError("huggingface_hub not installed.")

    client = InferenceClient(provider="hf-inference", api_key=cfg["token"])
    stream = client.chat.completions.create(model=cfg["model"], messages=messages, stream=True, max_tokens=1024)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _hf_complete(cfg: dict, messages: list) -> str:
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except ImportError:
        raise RuntimeError("huggingface_hub not installed.")

    client = InferenceClient(provider="hf-inference", api_key=cfg["token"])
    response = client.chat.completions.create(model=cfg["model"], messages=messages, stream=False, max_tokens=1024)
    return response.choices[0].message.content or ""


# =============================================================================
# OpenAI-compatible HTTP backend (raw httpx, for custom endpoints)
# =============================================================================

def _openai_complete(cfg: dict, messages: list) -> str:
    payload = {"model": cfg["model"], "messages": messages, "stream": False}
    with httpx.Client(timeout=cfg["timeout"]) as c:
        resp = c.post(cfg["endpoint"],
                      headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"},
                      json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _openai_stream(cfg: dict, messages: list) -> Generator[str, None, None]:
    payload = {"model": cfg["model"], "messages": messages, "stream": True}
    with httpx.Client(timeout=cfg["timeout"]) as c:
        with c.stream("POST", cfg["endpoint"],
                      headers={"Authorization": f"Bearer {cfg['token']}",
                                "Content-Type": "application/json",
                                "Accept": "text/event-stream"},
                      json=payload) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue


# =============================================================================
# Vertex AI backend
# =============================================================================

def _vertex_complete(cfg: dict, messages: list) -> str:
    payload = {"instances": [{"messages": messages}]}
    with httpx.Client(timeout=cfg["timeout"]) as c:
        resp = c.post(cfg["endpoint"],
                      headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"},
                      json=payload)
        resp.raise_for_status()
        pred = resp.json()["predictions"][0]
        return pred.get("content") or pred["candidates"][0]["content"]["parts"][0]["text"]


# =============================================================================
# Public API — blocking
# =============================================================================

def ask_medgemma(user_message: str,
                 image_bytes: Optional[bytes] = None,
                 history: Optional[List[dict]] = None) -> str:
    """Call the AI, return full reply. Never raises."""
    try:
        cfg      = _read_config()
        if cfg["style"] == "gemini":
            return _gemini_complete(cfg, user_message, image_bytes, history)
        
        messages = _build_messages(user_message, image_bytes, history)
        if cfg["style"] == "huggingface":
            return _hf_complete(cfg, messages)
        elif cfg["style"] == "vertex":
            return _vertex_complete(cfg, messages)
        else:
            return _openai_complete(cfg, messages)
    except Exception as exc:
        logger.error("Unexpected error calling AI: %s", exc)
        return _FALLBACK_REPLY


# =============================================================================
# Public API — streaming
# =============================================================================

def ask_medgemma_stream(user_message: str,
                        image_bytes: Optional[bytes] = None,
                        history: Optional[List[dict]] = None) -> Generator[str, None, None]:
    """Stream AI reply token-by-token. Yields friendly fallback on any error."""
    try:
        cfg = _read_config()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        yield _FALLBACK_REPLY
        return

    try:
        if cfg["style"] == "gemini":
            yield from _gemini_stream(cfg, user_message, image_bytes, history)
            return

        messages = _build_messages(user_message, image_bytes, history)
        if cfg["style"] == "vertex":
            yield _vertex_complete(cfg, messages)
        elif cfg["style"] == "openai":
            yield from _openai_stream(cfg, messages)
        else:
            yield from _hf_stream(cfg, messages)

    except httpx.TimeoutException:
        logger.error("Streaming request timed out.")
        yield "\n\n_(The AI stopped responding - please try again.)_"
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.error("HTTP %d during streaming.", status)
        if status in (401, 403):
            yield ("Authentication problem with the AI service. "
                   "Check your API key in backend/.env.")
        elif status == 429:
            yield "The AI service is busy (rate limit). Please wait and try again."
        else:
            yield _FALLBACK_REPLY
    except Exception as exc:
        logger.exception("Unexpected streaming error: %s", exc)
        yield _FALLBACK_REPLY


# =============================================================================
# Structured scan analysis
# =============================================================================

_ANALYZE_PROMPT = (
    "You are an MRI image quality expert. Examine this brain MRI image carefully and "
    "respond ONLY with a JSON object - no markdown, no explanation outside the JSON. "
    "Use exactly this structure:\n"
    "{\n"
    '  "artifact_present": true|false,\n'
    '  "artifact_type": "none"|"motion_blur"|"ghosting"|"banding"|"ringing"|"noise"|"other",\n'
    '  "severity": "none"|"mild"|"moderate"|"severe",\n'
    '  "quality_score": 0-100,\n'
    '  "region": "whole image"|"frontal"|"temporal"|"occipital"|"parietal"|"n/a",\n'
    '  "recommendation": "proceed"|"review"|"rescan",\n'
    '  "explanation": "One or two plain-English sentences a non-expert can understand."\n'
    "}\n"
    "Base your assessment purely on image quality, NOT on any clinical finding."
)

_ANALYZE_DEFAULT: dict = {
    "artifact_present": False,
    "artifact_type":    "none",
    "severity":         "none",
    "quality_score":    None,
    "region":           "n/a",
    "recommendation":   "review",
    "explanation": (
        "I'm sorry - I wasn't able to reach the AI model right now. "
        "Please check your internet connection and try again in a moment."
    ),
}


def analyze_scan(image_bytes: bytes) -> dict:
    """Return structured quality assessment dict. Never raises."""
    raw = ""
    try:
        raw = ask_medgemma(user_message=_ANALYZE_PROMPT, image_bytes=image_bytes)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(
                l for l in clean.splitlines() if not l.strip().startswith("```")
            ).strip()
        
        # Sometimes Gemini outputs ```json
        if clean.startswith("json"):
            clean = clean[4:].strip()
            
        parsed = json.loads(clean)
        result = dict(_ANALYZE_DEFAULT)
        result.update({k: v for k, v in parsed.items() if k in _ANALYZE_DEFAULT})
        qs = result.get("quality_score")
        if qs is not None:
            try:
                result["quality_score"] = max(0, min(100, int(qs)))
            except (ValueError, TypeError):
                result["quality_score"] = None
        return result
    except json.JSONDecodeError as exc:
        logger.warning("analyze_scan: JSON parse error (%s). Raw: %r", exc, raw[:200])
        result = dict(_ANALYZE_DEFAULT)
        result["explanation"] = raw[:200] if raw else _ANALYZE_DEFAULT["explanation"]
        return result
    except Exception as exc:
        logger.exception("analyze_scan: unexpected error: %s", exc)
        return dict(_ANALYZE_DEFAULT)
