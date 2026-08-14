import { useEffect, useRef, useState } from 'react'
import OptimizationDashboard from './optimization/OptimizationDashboard'

// FastAPI backend — override with VITE_API_URL if the server runs elsewhere.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function LogoMark() {
  return (
    <span className="logo-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 4l6 16 2-6 6-10" />
        <path d="M4 4h16" opacity="0.4" />
      </svg>
    </span>
  )
}

function Logo() {
  return (
    <a href="#top" className="logo" aria-label="ArmInferX home">
      <LogoMark />
      <span className="logo-text">
        Arm<strong>Infer</strong>X
      </span>
    </a>
  )
}

function Spinner() {
  return <span className="spinner" aria-hidden="true" />
}

function NavTabs({ view, onSelect }) {
  return (
    <nav className="site-nav" aria-label="Sections">
      <button
        type="button"
        className={`nav-tab${view === 'studio' ? ' is-active' : ''}`}
        onClick={() => onSelect('studio')}
        aria-current={view === 'studio' ? 'page' : undefined}
      >
        Studio
      </button>
      <button
        type="button"
        className={`nav-tab${view === 'optimization' ? ' is-active' : ''}`}
        onClick={() => onSelect('optimization')}
        aria-current={view === 'optimization' ? 'page' : undefined}
      >
        Optimization Dashboard
      </button>
    </nav>
  )
}

function summarizeDetail(detail) {
  // detail comes from res.json(), so it is always JSON-serializable.
  if (typeof detail === 'string') return detail
  return JSON.stringify(detail)
}

function formatLatency(latencyMs) {
  // Backend latency_ms is a non-negative number; anything else means no chip.
  if (typeof latencyMs !== 'number' || !Number.isFinite(latencyMs) || latencyMs < 0) return null
  const ms = Math.round(latencyMs) // round first so the unit threshold is exact
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`
  return `${ms} ms`
}

function describeError(err, res) {
  if (err instanceof TypeError) {
    return `Could not reach the backend at ${API_URL} — is the inference server running?`
  }
  if (res) {
    const detail = summarizeDetail(res.detail)
    return `Request failed (${res.status})${detail ? ` — ${detail}` : ''}`
  }
  return err?.message ?? 'Something went wrong while generating.'
}

// ---------------------------------------------------------------------------
// Engine selection (STEP 13)
//
// Presentation metadata for the selector. These are labels/device facts (model
// footprint from the validated report), not fabricated performance numbers.
// ---------------------------------------------------------------------------

const ENGINE_OPTIONS = [
  {
    id: 'llamacpp-optimized',
    label: 'llama.cpp — Q4_K_M',
    runtime: 'llama.cpp',
    device: 'CPU',
    supportsStreaming: true,
    context: 'Qwen2.5-0.5B-Instruct Q4_K_M · CPU-only · ~469 MB model footprint',
  },
]

async function postGenerate(payload) {
  const res = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  let body = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) {
    throw new Error(describeError(null, { status: res.status, detail: body?.detail }))
  }
  return body
}

// Streams POST /generate/stream (Server-Sent Events) and invokes onDelta(text)
// per token; resolves with the final done-event metadata (engine, latency,
// token counts, TTFT).
async function streamGenerate(payload, onDelta) {
  const res = await fetch(`${API_URL}/generate/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    let detail = null
    try {
      detail = (await res.json()).detail
    } catch {
      detail = null
    }
    throw new Error(
      typeof detail === 'string' && detail
        ? detail
        : `Request failed (${res.status})`,
    )
  }
  if (!res.body) throw new Error('Streaming is not supported by this browser')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let doneMeta = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const dataLine = block.split('\n').find((line) => line.startsWith('data:'))
      if (!dataLine) continue
      let ev
      try {
        ev = JSON.parse(dataLine.slice(5).trim())
      } catch {
        continue
      }
      if (ev.error) throw new Error(ev.error)
      if (ev.text) onDelta(ev.text)
      if (ev.done) doneMeta = ev
    }
  }
  if (!doneMeta) throw new Error('Stream ended without a done event')
  return doneMeta
}

function EngineSelector({ value, onChange, disabled }) {
  const selected = ENGINE_OPTIONS.find((opt) => opt.id === value) ?? ENGINE_OPTIONS[0]
  return (
    <div className="engine-select">
      <label className="engine-select-label" htmlFor="engine-select">
        Inference Engine
      </label>
      <div className="engine-select-row">
        <select
          id="engine-select"
          className="engine-select-input"
          value={selected.id}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
        >
          {ENGINE_OPTIONS.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
        <span className="engine-select-context mono">{selected.context}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Benchmark panel helpers
// ---------------------------------------------------------------------------

function splitLatency(latencyMs) {
  // Returns { value, unit } for a modern stat card, or null when unavailable.
  if (typeof latencyMs !== 'number' || !Number.isFinite(latencyMs) || latencyMs < 0) return null
  const ms = Math.round(latencyMs)
  if (ms >= 1000) return { value: (ms / 1000).toFixed(2), unit: 's' }
  return { value: String(ms), unit: 'ms' }
}

function formatMetric(value, digits = 1) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return value.toFixed(digits)
}

function formatTimestamp(iso) {
  if (!iso) return null
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return null
  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'medium' })
}

function MetricCard({ label, value, unit, time }) {
  return (
    <div className={`bench-card${time ? ' is-time' : ''}`}>
      <div className="bench-card-label">{label}</div>
      <div className="bench-card-value">
        {value == null ? <span className="bench-card-na">—</span> : value}
        {value != null && unit ? <span className="bench-card-unit">{unit}</span> : null}
      </div>
    </div>
  )
}

function formatTokens(value) {
  // Token counts are whole numbers (tokenizer-native output token IDs).
  return Number.isInteger(value) ? String(value) : null
}

function BenchmarkPanel({ benchmark, refreshing, onRefresh }) {
  const latency = benchmark ? splitLatency(benchmark.latency_ms) : null
  const ttft = benchmark ? splitLatency(benchmark.ttft_ms) : null
  const generatedTokens = benchmark ? formatTokens(benchmark.generated_tokens) : null
  const tokensPerSec = benchmark ? formatMetric(benchmark.tokens_per_second) : null
  const memory = benchmark ? formatMetric(benchmark.memory_mb) : null
  const cpu = benchmark ? formatMetric(benchmark.cpu_percent) : null
  const timestamp = benchmark ? formatTimestamp(benchmark.timestamp) : null
  const hasData = Boolean(
    latency ||
      ttft ||
      generatedTokens ||
      tokensPerSec != null ||
      memory != null ||
      cpu != null ||
      timestamp
  )

  return (
    <section className="bench-panel" aria-label="Benchmark metrics">
      <div className="bench-panel-head">
        <span className="bench-panel-title">
          <span className="bench-pulse" aria-hidden="true" />
          Benchmark · latest run
        </span>
        {benchmark?.engine_id && (
          <span className="bench-engine-chip mono">
            {benchmark.engine_id}
            {benchmark.runtime ? ` · ${benchmark.runtime}` : ''}
          </span>
        )}
        <button
          className={`bench-refresh${refreshing ? ' is-spinning' : ''}`}
          type="button"
          onClick={onRefresh}
          aria-label="Refresh benchmark data"
          title="Refresh benchmark data"
          disabled={refreshing}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 12a9 9 0 1 1-2.64-6.36" />
            <path d="M21 3v6h-6" />
          </svg>
        </button>
      </div>

      {hasData ? (
        <div className="bench-grid">
          <MetricCard label="Latency" value={latency?.value} unit={latency?.unit} />
          <MetricCard label="TTFT" value={ttft?.value} unit={ttft?.unit} />
          <MetricCard label="Generated Tokens" value={generatedTokens} />
          <MetricCard label="Tokens/sec" value={tokensPerSec} />
          <MetricCard label="Memory" value={memory} unit="MB" />
          <MetricCard label="CPU" value={cpu} unit="%" />
          <MetricCard label="Timestamp" value={timestamp} time />
        </div>
      ) : (
        <p className="bench-empty mono">
          No benchmark data yet — generate a response to record the first run.
        </p>
      )}
    </section>
  )
}

export default function App() {
  const [view, setView] = useState('studio')
  const [engineId, setEngineId] = useState('llamacpp-optimized')
  const [prompt, setPrompt] = useState('')
  const [conversation, setConversation] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [latestBenchmark, setLatestBenchmark] = useState(null)
  const [benchRefreshing, setBenchRefreshing] = useState(false)
  const textareaRef = useRef(null)
  const endRef = useRef(null)
  const itemIdRef = useRef(0)

  function nextItemId() {
    itemIdRef.current += 1
    return itemIdRef.current
  }

  useEffect(() => {
    autoResize()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [conversation, loading])

  function autoResize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }

  function handleChange(event) {
    setPrompt(event.target.value)
    autoResize()
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleGenerate(event)
    }
  }

  function handleSelectView(nextView) {
    setView(nextView)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Fetches the most recently saved benchmark record. Called automatically
  // after every inference; never throws so the chat flow is unaffected.
  async function refreshLatestBenchmark() {
    setBenchRefreshing(true)
    try {
      const res = await fetch(`${API_URL}/benchmarks/latest`)
      if (!res.ok) return
      const record = await res.json()
      setLatestBenchmark(record)
    } catch {
      // Backend unavailable or older — keep whatever the panel already shows.
    } finally {
      setBenchRefreshing(false)
    }
  }

  function applyResult(itemId, body) {
    setConversation((prev) =>
      prev.map((it) =>
        it.id === itemId
          ? {
              ...it,
              response: body?.response ?? it.response,
              model: body?.model ?? it.model,
              latencyMs: body?.latency_ms ?? it.latencyMs,
              engineId: body?.engine_id ?? it.engineId,
              runtime: body?.runtime ?? it.runtime,
              tokens: body?.generated_tokens ?? it.tokens,
              tokensPerSec: body?.tokens_per_second ?? it.tokensPerSec,
              ttftMs: body?.ttft_ms ?? it.ttftMs,
              streaming: false,
            }
          : it,
      ),
    )
  }

  async function handleGenerate(event) {
    event.preventDefault()
    const text = prompt.trim()
    if (!text || loading) return

    const engine = ENGINE_OPTIONS.find((opt) => opt.id === engineId) ?? ENGINE_OPTIONS[0]
    const itemId = nextItemId()
    const payload = { prompt: text, engine_id: engine.id }

    setLoading(true)
    setError(null)
    setConversation((prev) => [
      ...prev,
      {
        id: itemId,
        prompt: text,
        response: '',
        model: '',
        latencyMs: null,
        engineId: engine.id,
        runtime: engine.runtime,
        tokens: null,
        tokensPerSec: null,
        ttftMs: null,
        streaming: engine.supportsStreaming,
      },
    ])

    try {
      if (engine.supportsStreaming) {
        try {
          const meta = await streamGenerate(payload, (delta) => {
            setConversation((prev) =>
              prev.map((it) =>
                it.id === itemId ? { ...it, response: it.response + delta } : it,
              ),
            )
          })
          applyResult(itemId, meta)
        } catch (streamErr) {
          // Streaming unavailable/failed for this engine — fall back to the
          // non-streaming /generate so the chat still works.
          applyResult(itemId, await postGenerate(payload))
        }
      } else {
        applyResult(itemId, await postGenerate(payload))
      }
      refreshLatestBenchmark()
      // Only clear the composer after a successful generation, so a failed
      // request never discards the user's typed prompt.
      setPrompt('')
      autoResize()
    } catch (err) {
      setConversation((prev) =>
        prev.map((it) => (it.id === itemId ? { ...it, streaming: false } : it)),
      )
      setError(describeError(err, null))
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = prompt.trim().length > 0 && !loading

  return (
    <div className="app" id="top">
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      {/* Background decorations */}
      <div className="bg-grid" aria-hidden="true" />
      <div className="bg-glow bg-glow-a" aria-hidden="true" />

      {/* Header */}
      <header className="site-header">
        <div className="container header-inner">
          <Logo />
          <NavTabs view={view} onSelect={handleSelectView} />
          <span className="header-status mono">API · {API_URL}</span>
        </div>
      </header>

      <main id="main">
        {view === 'optimization' ? (
          <OptimizationDashboard />
        ) : (
        <section className="studio container" aria-labelledby="studio-title">
          <div className="studio-head">
            <p className="studio-kicker mono">AI Inference Studio</p>
            <h1 id="studio-title" className="studio-title">
              Generate with the local Q4_K_M model
            </h1>
            <p className="studio-sub">
              Prompt the model running on this machine. Responses appear as
              cards with the generated text and inference latency.
            </p>
          </div>

          {/* Composer */}
          <form className="composer" onSubmit={handleGenerate}>
            <label className="sr-only" htmlFor="prompt">
              Prompt
            </label>
            <textarea
              id="prompt"
              ref={textareaRef}
              rows={4}
              value={prompt}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything — e.g. “Explain Artificial Intelligence”"
              disabled={loading}
            />
            <EngineSelector value={engineId} onChange={setEngineId} disabled={loading} />
            <div className="composer-actions">
              <span className="composer-hint mono">
                Enter ↵ to generate · Shift+Enter for a new line
              </span>
              <button className="btn btn-primary" type="submit" disabled={!canSubmit}>
                {loading ? (
                  <>
                    <Spinner />
                    Generating…
                  </>
                ) : (
                  'Generate'
                )}
              </button>
            </div>
          </form>

          {error && (
            <div className="studio-error" role="alert">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 8v4" />
                <path d="M12 16h.01" />
              </svg>
              <span>{error}</span>
            </div>
          )}

          {/* Response area */}
          <section
            className="conversation"
            aria-label="Generation history"
            aria-live="polite"
            aria-busy={loading}
          >
            {conversation.length === 0 && !loading && (
              <div className="conversation-empty">
                <p className="mono">Your generations will appear here — prompt and response.</p>
              </div>
            )}

            {conversation.map((item) => {
              const latencyLabel = formatLatency(item.latencyMs)
              const ttftLabel = formatLatency(item.ttftMs)
              const hasMetadata =
                item.engineId ||
                item.runtime ||
                item.tokens != null ||
                item.tokensPerSec != null ||
                item.ttftMs != null
              return (
                <div className="exchange" key={item.id}>
                  <div className="msg msg-user">
                    <div className="msg-meta">You</div>
                    <div className="msg-bubble">{item.prompt}</div>
                  </div>
                  <div className="msg msg-ai">
                    <div className="msg-meta">
                      <LogoMark />
                      <span>{item.model || item.engineId || 'model'}</span>
                      {item.runtime && <span className="runtime-chip mono">{item.runtime}</span>}
                      {latencyLabel && <span className="latency-chip">{latencyLabel}</span>}
                    </div>
                    <div className="msg-bubble">
                      {item.response || (item.streaming ? (
                        <span className="typing-dots" aria-label="Generating response">
                          <span />
                          <span />
                          <span />
                        </span>
                      ) : null)}
                    </div>
                    {hasMetadata && (
                      <div className="meta-chips">
                        {item.engineId && (
                          <span className="meta-chip"><b>Engine</b>{item.engineId}</span>
                        )}
                        {item.runtime && (
                          <span className="meta-chip"><b>Runtime</b>{item.runtime}</span>
                        )}
                        {item.model && (
                          <span className="meta-chip"><b>Model</b>{item.model}</span>
                        )}
                        {latencyLabel && (
                          <span className="meta-chip"><b>Latency</b>{latencyLabel}</span>
                        )}
                        {item.tokens != null && (
                          <span className="meta-chip"><b>Tokens</b>{item.tokens}</span>
                        )}
                        {item.tokensPerSec != null && (
                          <span className="meta-chip"><b>Tokens/sec</b>{item.tokensPerSec}</span>
                        )}
                        {ttftLabel && (
                          <span className="meta-chip"><b>TTFT</b>{ttftLabel}</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}

            {loading && !conversation.some((item) => item.streaming) && (
              <div className="msg msg-ai">
                <div className="msg-meta">
                  <LogoMark />
                  <span>Thinking…</span>
                </div>
                <div className="msg-bubble">
                  <span className="typing-dots" aria-label="Generating response">
                    <span />
                    <span />
                    <span />
                  </span>
                </div>
              </div>
            )}

            {conversation.length > 0 && (
              <BenchmarkPanel
                benchmark={latestBenchmark}
                refreshing={benchRefreshing}
                onRefresh={refreshLatestBenchmark}
              />
            )}

            <div ref={endRef} />
          </section>
        </section>
        )}
      </main>

      {/* Footer */}
      <footer className="site-footer">
        <div className="container footer-inner">
          <span className="footer-tagline">
            ArmInferX · AI Inference Optimization Studio for Arm64 Cloud
          </span>
        </div>
      </footer>
    </div>
  )
}
