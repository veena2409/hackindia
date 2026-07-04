/**
 * api.js
 * Thin API client for the MedGemma Brain MRI Chat backend.
 *
 * sendMessage(sessionId, text, imageFile?) → async generator of string chunks
 * newSession()                             → Promise<string>  (session_id)
 * getDisclaimer()                          → Promise<string>  (disclaimer text)
 */

// In development (npm run dev): VITE_API_URL = http://localhost:8000  (set in .env.local)
// In production build:          VITE_API_URL is unset → empty string → same-origin
const BASE = import.meta.env.VITE_API_URL ?? "";
const TIMEOUT_MS = 120_000; // 2 min — MedGemma can be slow


// ── helpers ──────────────────────────────────────────────────────────────────

function makeAbortController(ms) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort("timeout"), ms);
  return { ctrl, cleanup: () => clearTimeout(timer) };
}

// ── public API ────────────────────────────────────────────────────────────────

/**
 * Streams the AI reply, yielding text chunks as they arrive.
 * Throws a user-friendly Error on network / timeout failure.
 *
 * @param {string}    sessionId
 * @param {string}    text
 * @param {File|null} imageFile
 * @returns {AsyncGenerator<string>}
 */
export async function* sendMessage(sessionId, text, imageFile = null) {
  const { ctrl, cleanup } = makeAbortController(TIMEOUT_MS);

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("message", text);
  if (imageFile) form.append("image", imageFile);

  let response;
  try {
    response = await fetch(`${BASE}/chat`, {
      method: "POST",
      body: form,
      signal: ctrl.signal,
    });
  } catch (err) {
    cleanup();
    if (err.name === "AbortError" || String(err).includes("timeout")) {
      throw new Error(
        "The AI is taking a while to respond. Please try again in a moment."
      );
    }
    throw new Error(
      "I couldn't reach the server — please check your connection and try again."
    );
  }

  if (!response.ok) {
    cleanup();
    throw new Error(
      `Server returned an error (${response.status}). Please try again.`
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      if (chunk) yield chunk;
    }
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("The response was cut short. Please try again.");
    }
    throw new Error("Something went wrong while reading the response.");
  } finally {
    cleanup();
    reader.releaseLock();
  }
}

/**
 * Create a new chat session on the backend.
 * @returns {Promise<string>} session_id
 */
export async function newSession() {
  const { ctrl, cleanup } = makeAbortController(10_000);
  try {
    const res = await fetch(`${BASE}/new_session`, {
      method: "POST",
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.session_id;
  } catch {
    throw new Error(
      "Couldn't start a new session — is the backend running?"
    );
  } finally {
    cleanup();
  }
}

/**
 * Fetch the medical disclaimer text.
 * @returns {Promise<string>}
 */
export async function getDisclaimer() {
  const { ctrl, cleanup } = makeAbortController(8_000);
  try {
    const res = await fetch(`${BASE}/disclaimer`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.disclaimer ?? "";
  } catch {
    return (
      "This is a learning tool, not a medical diagnosis. " +
      "Always consult a qualified professional about a real scan."
    );
  } finally {
    cleanup();
  }
}

/**
 * Upload an image to /analyze and return the structured quality assessment.
 * Includes heatmap_base64, artifact info, severity, quality_score, etc.
 * Never throws — returns null on any failure so the caller can hide the card.
 *
 * @param {File} imageFile
 * @returns {Promise<object|null>}
 */
export async function analyzeImage(imageFile) {
  const { ctrl, cleanup } = makeAbortController(90_000); // analysis can be slow
  try {
    const form = new FormData();
    form.append("image", imageFile);
    const res = await fetch(`${BASE}/analyze`, {
      method: "POST",
      body: form,
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("analyzeImage failed (non-fatal):", err.message);
    return null;
  } finally {
    cleanup();
  }
}

