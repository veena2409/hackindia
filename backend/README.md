# MedGemma Brain MRI Chat - Backend

A FastAPI backend (Python 3.10+) that wraps a MedGemma multimodal API to assess brain MRI image quality and chat about it.

---

## Project structure

```
backend/
  medgemma_client.py   - standalone MedGemma API client (this phase)
  requirements.txt     - Python dependencies
  .env.example         - copy to .env and fill in your credentials
  .env                 - your real credentials (never commit this!)
  .deps/               - pip packages installed here
  setup.bat            - one-click Windows setup
```

---

## Setup (Windows)

### Option A - Batch script (easiest)
```
cd backend
setup.bat
```

### Option B - Manual
```powershell
# Install dependencies
python -m pip install -r backend\requirements.txt --target backend\.deps

# Copy env template
copy backend\.env.example backend\.env
```

Then open backend\.env and fill in the three required values:

| Variable               | Description                                      |
|------------------------|--------------------------------------------------|
| MEDGEMMA_ENDPOINT_URL  | Full URL of your MedGemma inference endpoint     |
| MEDGEMMA_AUTH_TOKEN    | Bearer token / API key                           |
| MEDGEMMA_MODEL         | Model ID (e.g. medgemma-3-4b-it)                |
| MEDGEMMA_API_STYLE     | openai (default) or vertex                       |
| MEDGEMMA_TIMEOUT       | Request timeout in seconds (default 120)         |

---

## Running the smoke test

```powershell
# Set PYTHONPATH so Python finds packages in .deps
$env:PYTHONPATH = "backend\.deps"

# Text-only test
python backend\medgemma_client.py

# Image + text test (replace with your MRI file)
python backend\medgemma_client.py path\to\scan.png

# Streaming image test
python backend\medgemma_client.py path\to\scan.png --stream
```

### Expected output with a real endpoint
```
============================================================
  MedGemma Client -- Smoke Test
============================================================

[Test 1] Text-only question
Q: What kinds of motion artifacts can appear in a brain MRI?
A: Great question! Motion artifacts in brain MRIs can show up in a few ways...
  2.34s

[Test 2] Image + text question
Loaded image: scan.png  (543,210 bytes)
Q: How is the quality of this scan?
A: This scan looks pretty clear overall! I can see...
  3.87s

============================================================
  Smoke test complete.
============================================================
```

### Expected output with placeholder credentials
The client gracefully returns a friendly fallback - no crash, no raw exception shown to the user.

---

## Public API

```python
from medgemma_client import ask_medgemma, ask_medgemma_stream

# Synchronous
reply = ask_medgemma(
    user_message="How is the quality of this scan?",
    image_bytes=open("scan.png", "rb").read(),   # optional
    history=[                                      # optional
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
    ],
)

# Streaming (yields str chunks)
for chunk in ask_medgemma_stream("Explain the blurring.", image_bytes=img):
    print(chunk, end="", flush=True)
```

---

## Medical safety note

MedGemma is NOT a clinical device. This tool is an educational image-quality helper only.
The system prompt instructs the model to:
- Never state a diagnosis
- Always remind users that a qualified professional must review the scan
- Use calm, reassuring language
