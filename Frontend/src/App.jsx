import { useState, useReducer, useEffect, useRef, useCallback } from 'react';
import './index.css';
import { sendMessage, newSession, getDisclaimer, analyzeImage } from './api';

// ── State ─────────────────────────────────────────────────────────────────────

const initialState = {
  sessionId: null,
  messages: [],
  streaming: false,
  pendingImage: null, // { file, url }
  scanCard: null,     // current analysis result or 'loading'
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_SESSION':
      return { ...state, sessionId: action.id };

    case 'RESET':
      return { ...initialState, sessionId: action.id };

    case 'ADD_MSG':
      return { ...state, messages: [...state.messages, action.msg] };

    case 'APPEND_AI': {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === 'ai' && last.id === action.id) {
        msgs[msgs.length - 1] = { ...last, text: last.text + action.chunk };
      }
      return { ...state, messages: msgs };
    }

    case 'SET_STREAMING':
      return { ...state, streaming: action.val };

    case 'SET_PENDING_IMAGE':
      return { ...state, pendingImage: action.img };

    case 'CLEAR_PENDING_IMAGE':
      return { ...state, pendingImage: null };

    case 'SET_SCAN_CARD':
      return { ...state, scanCard: action.data };

    default:
      return state;
  }
}

// ── Quick-reply chips ─────────────────────────────────────────────────────────

const CHIPS = [
  'Is this scan clear?',
  'Is there motion blur?',
  'Should this be re-taken?',
];

function uid() {
  return Math.random().toString(36).slice(2);
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [input, setInput] = useState('');
  const [disclaimer, setDisclaimer] = useState('');
  const [disclaimerVisible, setDisclaimerVisible] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // Always-current ref for sessionId so callbacks never capture stale value
  const sessionIdRef = useRef(null);
  const chatBodyRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const dragCounterRef = useRef(0);
  const streamingRef = useRef(false);
  const pendingImageRef = useRef(null);

  const { sessionId, messages, streaming, pendingImage, scanCard } = state;

  // Keep refs in sync
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  useEffect(() => { streamingRef.current = streaming; }, [streaming]);
  useEffect(() => { pendingImageRef.current = pendingImage; }, [pendingImage]);

  // ── Init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const [sid, disc] = await Promise.all([newSession(), getDisclaimer()]);
        sessionIdRef.current = sid;
        dispatch({ type: 'SET_SESSION', id: sid });
        if (disc) { setDisclaimer(disc); setDisclaimerVisible(true); }
      } catch (e) {
        console.error('Init failed:', e);
      }
    })();
  }, []);

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    const el = chatBodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming, scanCard]);

  // ── Textarea auto-resize ──────────────────────────────────────────────────
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
  }, [input]);

  // ── Core send — reads from refs, never stale ─────────────────────────────
  const handleSend = useCallback(async (text, imageFileArg) => {
    const sid = sessionIdRef.current;
    const isStreaming = streamingRef.current;
    const pImg = pendingImageRef.current;

    if (!text.trim() && !imageFileArg && !pImg) return;
    if (isStreaming || !sid) return;

    const imageFile = imageFileArg ?? pImg?.file ?? null;
    const imageUrl  = imageFileArg
      ? URL.createObjectURL(imageFileArg)
      : (pImg?.url ?? null);

    const userMsg = { id: uid(), role: 'user', text: text.trim(), imageUrl };
    dispatch({ type: 'ADD_MSG', msg: userMsg });
    dispatch({ type: 'SET_STREAMING', val: true });
    dispatch({ type: 'CLEAR_PENDING_IMAGE' });
    setInput('');

    const aiId = uid();
    dispatch({ type: 'ADD_MSG', msg: { id: aiId, role: 'ai', text: '' } });

    // Fire /analyze in background if there's an image
    if (imageFile) {
      dispatch({ type: 'SET_SCAN_CARD', data: 'loading' });
      analyzeImage(imageFile)
        .then(result => dispatch({ type: 'SET_SCAN_CARD', data: result }))
        .catch(() => dispatch({ type: 'SET_SCAN_CARD', data: null }));
    }

    try {
      let got = false;
      for await (const chunk of sendMessage(sid, text.trim(), imageFile)) {
        got = true;
        dispatch({ type: 'APPEND_AI', id: aiId, chunk });
      }
      if (!got) {
        dispatch({ type: 'APPEND_AI', id: aiId, chunk: '(No response received.)' });
      }
    } catch (err) {
      dispatch({
        type: 'ADD_MSG',
        msg: {
          id: uid(), role: 'error',
          text: err.message || "I couldn't reach the server — please try again.",
        },
      });
    } finally {
      dispatch({ type: 'SET_STREAMING', val: false });
      textareaRef.current?.focus();
    }
  }, []); // no deps — reads everything from refs

  // ── Keyboard ──────────────────────────────────────────────────────────────
  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(input, null);
    }
  };

  // ── Image staging ─────────────────────────────────────────────────────────
  const stageImage = useCallback((file) => {
    if (!file) return;
    if (pendingImageRef.current?.url) URL.revokeObjectURL(pendingImageRef.current.url);
    dispatch({ type: 'SET_PENDING_IMAGE', img: { file, url: URL.createObjectURL(file) } });
  }, []);

  const handleFileChange = (e) => { stageImage(e.target.files?.[0]); e.target.value = ''; };

  // ── Drag & drop ───────────────────────────────────────────────────────────
  const handleDragEnter = (e) => { e.preventDefault(); dragCounterRef.current += 1; setDragOver(true); };
  const handleDragLeave = (e) => {
    e.preventDefault();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current <= 0) { dragCounterRef.current = 0; setDragOver(false); }
  };
  const handleDragOver  = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault(); dragCounterRef.current = 0; setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) stageImage(file);
  };

  // ── New chat ──────────────────────────────────────────────────────────────
  const handleNewChat = async () => {
    if (pendingImageRef.current?.url) URL.revokeObjectURL(pendingImageRef.current.url);
    try {
      const sid = await newSession();
      sessionIdRef.current = sid;
      dispatch({ type: 'RESET', id: sid });
      setInput('');
    } catch {
      dispatch({ type: 'RESET', id: sessionIdRef.current });
    }
  };

  // ── Sample scan ───────────────────────────────────────────────────────────
  const handleSampleScan = async () => {
    if (streamingRef.current) return;
    // If session not yet created, wait briefly
    if (!sessionIdRef.current) {
      await new Promise(r => setTimeout(r, 800));
      if (!sessionIdRef.current) return; // still not ready, bail gracefully
    }
    try {
      const res = await fetch('/samples/sample_brain_mri.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'sample_brain_mri.jpg', { type: 'image/jpeg' });
      await handleSend('How is the quality of this scan?', file);
    } catch {
      dispatch({
        type: 'ADD_MSG',
        msg: { id: uid(), role: 'error', text: "Couldn't load the sample image." },
      });
    }
  };

  // ── Chip ─────────────────────────────────────────────────────────────────
  const handleChip = (text) => { setInput(text); textareaRef.current?.focus(); };

  // ── Render ────────────────────────────────────────────────────────────────
  const canSend = !streaming && sessionId && (input.trim() || pendingImage);

  return (
    <div
      className="app"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">🧠</div>
          <div>
            <div className="header-title">MRI Quality Assistant</div>
            <div className="header-subtitle">Powered by MedGemma · Educational use only</div>
          </div>
        </div>
        <div className="header-actions">
          <button className="btn-ghost" onClick={handleNewChat}>✦ New chat</button>
        </div>
      </header>

      {/* Disclaimer */}
      {disclaimerVisible && (
        <div className="disclaimer" role="alert">
          <span className="disclaimer-icon">ℹ️</span>
          <p className="disclaimer-text">{disclaimer}</p>
          <button className="disclaimer-close" onClick={() => setDisclaimerVisible(false)} aria-label="Dismiss">×</button>
        </div>
      )}

      {/* Scan Analysis Card */}
      {scanCard && (
        <ScanCard data={scanCard} onClose={() => dispatch({ type: 'SET_SCAN_CARD', data: null })} />
      )}

      {/* Chat body */}
      <main className="chat-body" ref={chatBodyRef}>
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">🔬</div>
            <h1>Upload a brain MRI image and ask me anything about it — I'll tell you if the scan looks clear.</h1>
            <p>I can describe image quality, spot motion blur, and answer your questions in plain language.</p>
            <button className="sample-btn" onClick={handleSampleScan} disabled={streaming}>
              🖼 Try a sample scan
            </button>
          </div>
        )}

        {messages.map((msg) => (
          <MessageRow key={msg.id} msg={msg} streaming={streaming} messages={messages} />
        ))}

        {streaming && messages[messages.length - 1]?.role === 'ai' && messages[messages.length - 1]?.text === '' && (
          <div className="message-row ai">
            <div className="avatar">🤖</div>
            <div className="bubble ai"><TypingIndicator /></div>
          </div>
        )}
      </main>

      {/* Input area */}
      <div className="input-area">
        <div className="chips">
          {CHIPS.map((c) => (
            <button key={c} className="chip" onClick={() => handleChip(c)} disabled={streaming}>{c}</button>
          ))}
        </div>

        {pendingImage && (
          <div className="image-preview-chip">
            <img src={pendingImage.url} alt="staged MRI" className="preview-thumb" />
            <span>MRI image ready</span>
            <button className="remove-img" onClick={() => dispatch({ type: 'CLEAR_PENDING_IMAGE' })} aria-label="Remove">×</button>
          </div>
        )}

        {streaming && (
          <div className="streaming-status" aria-live="polite">
            <span className="typing-dot" style={{ width: 5, height: 5 }} />
            AI is looking at your scan…
          </div>
        )}

        <div className="compose">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={pendingImage ? 'Ask about this scan…' : 'Type a message or drop an MRI image here…'}
            rows={1}
            disabled={streaming}
            aria-label="Chat message"
            id="chat-input"
          />
          <div className="compose-actions">
            <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} id="file-input" />
            <button className="upload-btn" onClick={() => fileInputRef.current?.click()} disabled={streaming} title="Add MRI image" aria-label="Add MRI image">＋</button>
            <button className="send-btn" onClick={() => handleSend(input, null)} disabled={!canSend} aria-label="Send" title="Send (Enter)">➤</button>
          </div>
        </div>
      </div>

      {dragOver && (
        <div className="drag-overlay">
          <div className="drag-overlay-inner">
            <div className="icon">🧠</div>
            <p>Drop your MRI image here</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ── MessageRow ────────────────────────────────────────────────────────────────

function MessageRow({ msg, streaming, messages }) {
  const isLastAi = msg.role === 'ai' && msg === messages[messages.length - 1];
  const isStreaming = streaming && isLastAi;

  return (
    <div className={`message-row ${msg.role}`}>
      {msg.role === 'ai'    && <div className="avatar">🤖</div>}
      {msg.role === 'error' && <div className="avatar" style={{ background: '#ffd6d6' }}>⚠️</div>}
      <div className={`bubble ${msg.role}`}>
        {msg.imageUrl && <img src={msg.imageUrl} alt="Uploaded MRI" className="bubble-thumb" />}
        <span style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</span>
        {isStreaming && msg.text && (
          <span style={{
            display: 'inline-block', width: 2, height: '1em',
            background: 'var(--clr-accent)', marginLeft: 2, verticalAlign: 'text-bottom',
            animation: 'blink 0.9s infinite',
          }} />
        )}
      </div>
    </div>
  );
}

// ── TypingIndicator ───────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" />
    </div>
  );
}

// ── ScanCard ──────────────────────────────────────────────────────────────────

const SEVERITY_COLORS = {
  none:     { bg: '#e8f5e9', text: '#2e7d32', border: '#a5d6a7', label: 'None' },
  mild:     { bg: '#fff9c4', text: '#f57f17', border: '#fff176', label: 'Mild' },
  moderate: { bg: '#fff3e0', text: '#e65100', border: '#ffcc80', label: 'Moderate' },
  severe:   { bg: '#ffebee', text: '#c62828', border: '#ef9a9a', label: 'Severe' },
};

const REC_CONFIG = {
  proceed: { icon: '✅', color: '#2e7d32', bg: '#e8f5e9', label: 'Proceed — image quality is acceptable' },
  review:  { icon: '🔎', color: '#1565c0', bg: '#e3f2fd', label: 'Review Recommended — check with a professional' },
  rescan:  { icon: '🔄', color: '#b71c1c', bg: '#ffebee', label: 'Rescan Recommended — image quality may be insufficient' },
};

function QualityGauge({ score }) {
  if (score == null) {
    return <div className="gauge-na">N/A</div>;
  }
  const color = score > 70 ? '#2e7d32' : score >= 40 ? '#f57f17' : '#c62828';
  const r = 28, circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  return (
    <div className="gauge-wrap">
      <svg width="72" height="72" viewBox="0 0 72 72">
        <circle cx="36" cy="36" r={r} fill="none" stroke="#e0e0e0" strokeWidth="7" />
        <circle
          cx="36" cy="36" r={r} fill="none"
          stroke={color} strokeWidth="7"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 36 36)"
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
        <text x="36" y="40" textAnchor="middle" fontSize="14" fontWeight="700" fill={color}>{score}</text>
      </svg>
      <span className="gauge-label" style={{ color }}>Quality</span>
    </div>
  );
}

function ScanCard({ data, onClose }) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapLoaded, setHeatmapLoaded] = useState(false);

  if (data === 'loading') {
    return (
      <div className="scan-card scan-card--loading">
        <div className="scan-card-header">
          <span className="scan-card-title">🔬 Analyzing scan…</span>
        </div>
        <div className="skeleton-row">
          <div className="skeleton skeleton-gauge" />
          <div className="skeleton-lines">
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line short" />
            <div className="skeleton skeleton-line" />
          </div>
        </div>
      </div>
    );
  }

  const sev = SEVERITY_COLORS[data.severity] ?? SEVERITY_COLORS.none;
  const rec = REC_CONFIG[data.recommendation] ?? REC_CONFIG.review;
  const hasHeatmap = !!data.heatmap_base64;

  return (
    <div className="scan-card">
      <div className="scan-card-header">
        <span className="scan-card-title">🔬 Scan Analysis</span>
        <button className="scan-card-close" onClick={onClose} aria-label="Close analysis">×</button>
      </div>

      {/* Recommendation banner */}
      <div className="rec-banner" style={{ background: rec.bg, color: rec.color }}>
        <span className="rec-icon">{rec.icon}</span>
        <span>{rec.label}</span>
      </div>

      <div className="scan-card-body">
        {/* Gauge */}
        <QualityGauge score={data.quality_score} />

        {/* Details */}
        <div className="scan-details">
          <div className="scan-detail-row">
            <span className="detail-label">Artifact</span>
            <span className="detail-value">{data.artifact_type?.replace('_', ' ') || 'None'}</span>
          </div>
          <div className="scan-detail-row">
            <span className="detail-label">Severity</span>
            <span
              className="severity-badge"
              style={{ background: sev.bg, color: sev.text, border: `1px solid ${sev.border}` }}
            >{sev.label}</span>
          </div>
          {data.region && data.region !== 'n/a' && (
            <div className="scan-detail-row">
              <span className="detail-label">Region</span>
              <span className="detail-value">{data.region}</span>
            </div>
          )}
        </div>
      </div>

      {/* Explanation */}
      <p className="scan-explanation">{data.explanation}</p>

      {/* Heatmap section */}
      {hasHeatmap && (
        <div className="heatmap-section">
          <div className="heatmap-toggle-row">
            <span className="heatmap-toggle-label">Image Sharpness Map</span>
            <button
              className={`toggle-btn ${showHeatmap ? 'active' : ''}`}
              onClick={() => setShowHeatmap(v => !v)}
            >
              {showHeatmap ? 'Hide map' : 'Show map'}
            </button>
          </div>

          {showHeatmap && (
            <div className="heatmap-container">
              {!heatmapLoaded && <div className="skeleton" style={{ height: 180, borderRadius: 8 }} />}
              <img
                src={`data:image/png;base64,${data.heatmap_base64}`}
                alt="Image sharpness heatmap"
                className="heatmap-img"
                style={{ display: heatmapLoaded ? 'block' : 'none' }}
                onLoad={() => setHeatmapLoaded(true)}
              />
              {heatmapLoaded && (
                <>
                  {/* Color legend */}
                  <div className="heatmap-legend">
                    <div className="legend-gradient" />
                    <div className="legend-labels">
                      <span>Blurry</span>
                      <span>Sharp</span>
                    </div>
                  </div>
                  <p className="heatmap-caption">
                    Image Sharpness Map — warmer areas are sharper, cooler areas show more blur/motion.
                    This is a quality visualization, not a diagnostic map.
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      )}

      <p className="scan-disclaimer">
        ⚠️ Educational tool only — not a medical diagnosis. Always consult a qualified professional.
      </p>
    </div>
  );
}
