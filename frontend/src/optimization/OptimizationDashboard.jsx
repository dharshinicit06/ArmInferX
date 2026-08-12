import { useCallback, useEffect, useState } from 'react'
import {
  loadOptimizationReport,
  formatNumber,
  NOT_MEASURED,
} from './loadReport'
import './OptimizationDashboard.css'

// ---------------------------------------------------------------------------
// Small presentational pieces
// ---------------------------------------------------------------------------

function SectionHead({ kicker, title, sub }) {
  return (
    <div className="dash-sec-head">
      {kicker && <p className="dash-sec-kicker mono">{kicker}</p>}
      <h2 className="dash-sec-title">{title}</h2>
      {sub && <p className="dash-sec-sub">{sub}</p>}
    </div>
  )
}

function StatusBadge({ tone, children }) {
  return (
    <span className={`status-badge status-${tone}`}>
      <span className="status-dot" aria-hidden="true" />
      {children}
    </span>
  )
}

function Value({ value, unit, className }) {
  if (value == null) {
    return <span className="value-na">Not measured</span>
  }
  return (
    <span className={className}>
      {value}
      {unit ? <span className="unit">{unit}</span> : null}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Section 2 — Optimization summary metric cards
// ---------------------------------------------------------------------------

function SummaryGrid({ items }) {
  return (
    <div className="metric-grid">
      {items.map((item) => (
        <div className="metric-card" key={item.key}>
          <div className="metric-label">{item.label}</div>
          <div className="metric-value">
            <Value value={item.value} unit={item.unit} />
          </div>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 3 — Model footprint comparison
// ---------------------------------------------------------------------------

function FootprintSection({ footprint }) {
  const fp16Mb = footprint.fp16.mb
  const q4Mb = footprint.q4K.mb
  const reduction = footprint.reductionPercent

  // Scale bars relative to the larger model so the visual gap is honest.
  const maxMb = Math.max(fp16Mb ?? 0, q4Mb ?? 0) || 1
  const fp16Width = fp16Mb == null ? 0 : Math.max((fp16Mb / maxMb) * 100, 4)
  const q4Width = q4Mb == null ? 0 : Math.max((q4Mb / maxMb) * 100, 4)

  return (
    <section className="dash-section" aria-labelledby="footprint-title">
      <SectionHead
        kicker="Model footprint"
        title="Storage Footprint Reduction"
        sub="Q4_K_M reduces model storage footprint compared with FP16. This does not represent a measured inference-speed improvement."
      />
      <div className="footprint">
        <div className="footprint-rows">
          <div className="footprint-row">
            <div className="footprint-meta">
              <span className="footprint-name">FP16 (reference)</span>
              <span className="footprint-size mono">
                {fp16Mb == null ? NOT_MEASURED : `~${formatNumber(fp16Mb, 1)} MB`}
              </span>
            </div>
            <div className="footprint-track">
              <div
                className="footprint-bar bar-fp16"
                style={{ width: `${fp16Width}%` }}
              />
            </div>
          </div>
          <div className="footprint-row">
            <div className="footprint-meta">
              <span className="footprint-name">Q4_K_M (validated)</span>
              <span className="footprint-size mono">
                {q4Mb == null ? NOT_MEASURED : `~${formatNumber(q4Mb, 1)} MB`}
              </span>
            </div>
            <div className="footprint-track">
              <div
                className="footprint-bar bar-q4"
                style={{ width: `${q4Width}%` }}
              />
            </div>
          </div>
        </div>
        <div className="footprint-reduction">
          <span className="footprint-reduction-label">Storage reduction</span>
          <span className="footprint-reduction-value mono">
            {reduction == null ? NOT_MEASURED : `${formatNumber(reduction, 2)}%`}
          </span>
        </div>
      </div>
      {footprint.note && <p className="footprint-note">{footprint.note}</p>}
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section 4 — FP16 feasibility banner
// ---------------------------------------------------------------------------

function FeasibilityBanner({ feasibility }) {
  const statusTone = feasibility.inferenceCompleted ? 'ok' : 'danger'
  return (
    <section className="dash-section" aria-labelledby="feasibility-title">
      <SectionHead
        kicker="Feasibility"
        title="FP16 Baseline: Not Feasible on Current Machine"
        sub="The Transformers FP16 baseline could not complete model loading because the development machine has only 7.63 GiB RAM and entered severe memory pressure during the feasibility check."
      />
      <div className="feasibility">
        <div className="feasibility-stats">
          <div className="feasibility-stat">
            <span className="feasibility-stat-label">Total RAM</span>
            <span className="feasibility-stat-value mono">
              {feasibility.totalRamGb == null
                ? NOT_MEASURED
                : `${formatNumber(feasibility.totalRamGb, 2)} GiB`}
            </span>
          </div>
          <div className="feasibility-stat">
            <span className="feasibility-stat-label">Result</span>
            <span className="feasibility-stat-value">
              <StatusBadge tone={statusTone}>
                {feasibility.inferenceCompleted ? 'FEASIBLE' : 'NOT FEASIBLE'}
              </StatusBadge>
            </span>
          </div>
          <div className="feasibility-stat">
            <span className="feasibility-stat-label">FP16 inference</span>
            <span className="feasibility-stat-value">
              <StatusBadge tone="warn">NOT MEASURED</StatusBadge>
            </span>
          </div>
        </div>
        {feasibility.reason && (
          <p className="feasibility-reason">{feasibility.reason}</p>
        )}
        {feasibility.availableRamMb != null && (
          <p className="feasibility-detail mono">
            Watchdog trigger · available RAM at abort:{' '}
            <strong>{formatNumber(feasibility.availableRamMb, 0)} MB</strong>{' '}
            (below the severe-paging alarm)
          </p>
        )}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section 5 — Measured vs Not measured
// ---------------------------------------------------------------------------

function EvidenceList({ title, tone, items, empty }) {
  return (
    <div className={`evidence-col evidence-${tone}`}>
      <h3 className="evidence-title">{title}</h3>
      {items.length === 0 ? (
        <p className="evidence-empty">{empty}</p>
      ) : (
        <ul className="evidence-list">
          {items.map((item) => (
            <li key={item.label}>
              <span className="evidence-label">{item.label}</span>
              {item.value == null ? (
                <span className="evidence-value value-na">Not measured</span>
              ) : (
                <span className="evidence-value mono">
                  {item.value}
                  {item.unit ? <span className="unit">{item.unit}</span> : null}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function MeasuredVsNotMeasured({ measured, notMeasured }) {
  const notMeasuredList = notMeasured.map((item) => ({
    ...item,
    value: null, // force the "Not measured" presentation
  }))
  return (
    <section className="dash-section" aria-labelledby="evidence-title">
      <SectionHead
        kicker="Evidence boundaries"
        title="Measured vs Not Measured"
        sub="Unavailable metrics are intentionally not estimated."
      />
      <div className="evidence-grid">
        <EvidenceList
          title="Measured"
          tone="ok"
          items={measured}
          empty="No measured metrics available in the report."
        />
        <EvidenceList
          title="Not measured"
          tone="warn"
          items={notMeasuredList}
          empty="No unmeasured FP16 metrics declared in the report."
        />
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section 6 — Benchmark run history
// ---------------------------------------------------------------------------

function RunsTable({ runs, aggregates }) {
  if (runs.length === 0) {
    return (
      <div className="runs-fallback">
        <p>
          The report does not contain individual run records — aggregate
          benchmark information is shown instead.
        </p>
        <div className="runs-fallback-chips">
          <span className="chip mono">
            Runs · {aggregates.runs ?? NOT_MEASURED}
          </span>
          <span className="chip mono">
            Mean latency · {aggregates.meanLatencyMs == null ? NOT_MEASURED : formatNumber(aggregates.meanLatencyMs, 2)} ms
          </span>
          <span className="chip mono">
            Mean TTFT · {aggregates.meanTtftMs == null ? NOT_MEASURED : formatNumber(aggregates.meanTtftMs, 2)} ms
          </span>
          <span className="chip mono">
            Mean tokens/s · {aggregates.meanTokensPerSec == null ? NOT_MEASURED : formatNumber(aggregates.meanTokensPerSec, 2)}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="runs-table-wrap">
      <table className="runs-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Latency</th>
            <th>TTFT</th>
            <th>Tokens</th>
            <th>Tokens/sec</th>
            <th>Memory</th>
            <th>CPU</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run}>
              <td className="mono">{run.run}</td>
              <td className="mono">
                {run.latencyValue == null
                  ? NOT_MEASURED
                  : `${run.latencyValue} ${run.latencyUnit}`}
              </td>
              <td className="mono">
                {run.ttftValue == null
                  ? NOT_MEASURED
                  : `${run.ttftValue} ${run.ttftUnit}`}
              </td>
              <td className="mono">{run.tokens ?? NOT_MEASURED}</td>
              <td className="mono">{run.tokensPerSec ?? NOT_MEASURED}</td>
              <td className="mono">
                {run.memory == null ? NOT_MEASURED : `${run.memory} MB`}
              </td>
              <td className="mono">
                {run.cpu == null ? NOT_MEASURED : `${run.cpu} %`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RunHistory({ runs, aggregates }) {
  return (
    <section className="dash-section" aria-labelledby="runs-title">
      <SectionHead
        kicker="Run history"
        title="Benchmark Run History"
        sub="Five latest timed runs from the report (1 warmup excluded)."
      />
      <RunsTable runs={runs} aggregates={aggregates} />
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section 7 — Benchmark configuration
// ---------------------------------------------------------------------------

function ConfigurationGrid({ configuration }) {
  return (
    <section className="dash-section" aria-labelledby="config-title">
      <SectionHead
        kicker="Methodology"
        title="Benchmark Configuration"
        sub="Exact configuration recorded in the optimization report."
      />
      <div className="config-grid">
        {configuration.map((item) => (
          <div className="config-cell" key={item.label}>
            <span className="config-label">{item.label}</span>
            <span className="config-value mono">{item.value}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section 8 — Evidence & limitations
// ---------------------------------------------------------------------------

function LimitationsSection({ limitations }) {
  const fixed = [
    'The dashboard only displays metrics supported by the generated report.',
  ]
  const bullets = limitations.length > 0 ? limitations : fixed
  return (
    <section className="dash-section" aria-labelledby="limitations-title">
      <SectionHead
        kicker="Scope"
        title="Evidence & Limitations"
        sub="Measured facts and hardware constraints, kept separate."
      />
      <ul className="limitations-list">
        {bullets.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
        {limitations.length > 0 && <li>{fixed[0]}</li>}
      </ul>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Loading / error states
// ---------------------------------------------------------------------------

function DashboardLoading() {
  return (
    <div className="dash-state">
      <span className="spinner dash-spinner" aria-hidden="true" />
      <p className="dash-state-title">Loading optimization report…</p>
      <p className="dash-state-sub mono">Fetching measured benchmark evidence</p>
    </div>
  )
}

function DashboardError({ message, onRetry }) {
  return (
    <div className="dash-state" role="alert">
      <span className="dash-state-icon" aria-hidden="true">!</span>
      <p className="dash-state-title">Optimization report unavailable</p>
      <p className="dash-state-sub">
        {message ||
          'Could not load the report. Start the backend or bundle a fresh copy of the report.'}
      </p>
      <button className="btn btn-primary" type="button" onClick={onRetry}>
        Retry
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export default function OptimizationDashboard() {
  const [state, setState] = useState({ status: 'loading', report: null, source: null, error: null })

  const load = useCallback(async () => {
    setState((prev) => ({ ...prev, status: 'loading', error: null }))
    try {
      const { report, source } = await loadOptimizationReport()
      setState({ status: 'ready', report, source, error: null })
    } catch (err) {
      setState({
        status: 'error',
        report: null,
        source: null,
        error: err?.message || 'Failed to load the optimization report.',
      })
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  if (state.status === 'loading') return <DashboardLoading />
  if (state.status === 'error') return <DashboardError message={state.error} onRetry={load} />

  const { report, source } = state

  return (
    <div className="dash container-wide">
      {/* 1 — Header */}
      <header className="dash-hero">
        <p className="dash-hero-kicker mono">STEP 12 · Optimization Evidence</p>
        <h1 className="dash-hero-title">{report.meta.title}</h1>
        <p className="dash-hero-sub">{report.meta.subtitle}</p>
        <div className="dash-badges">
          <span className="dash-badge">
            <span className="dash-badge-label">Runtime</span>
            <span className="dash-badge-value">{report.meta.runtime}</span>
          </span>
          <span className="dash-badge">
            <span className="dash-badge-label">Engine</span>
            <span className="dash-badge-value">{report.meta.engineId}</span>
          </span>
          <span className="dash-badge">
            <span className="dash-badge-label">Model</span>
            <span className="dash-badge-value">{report.meta.modelLabel}</span>
          </span>
          <span className="dash-badge">
            <span className="dash-badge-label">Platform</span>
            <span className="dash-badge-value">{report.meta.platform}</span>
          </span>
          <span className="dash-badge dash-badge-status">
            <span className="dash-badge-label">Benchmark status</span>
            <StatusBadge tone="ok">{report.meta.benchmarkStatus}</StatusBadge>
          </span>
        </div>
      </header>

      {/* 2 — Summary */}
      <section className="dash-section" aria-labelledby="summary-title">
        <SectionHead
          kicker="Measured results"
          title="Optimization Summary"
          sub="Aggregate metrics from the Q4_K_M benchmark (5 timed runs, greedy decoding)."
        />
        <SummaryGrid items={report.summary} />
      </section>

      {/* 3 — Footprint */}
      <FootprintSection footprint={report.footprint} />

      {/* 4 — Feasibility */}
      <FeasibilityBanner feasibility={report.feasibility} />

      {/* 5 — Measured vs not measured */}
      <MeasuredVsNotMeasured
        measured={report.measured}
        notMeasured={report.notMeasured}
      />

      {/* 6 — Run history */}
      <RunHistory runs={report.runs} aggregates={report.runAggregates} />

      {/* 7 — Configuration */}
      <ConfigurationGrid configuration={report.configuration} />

      {/* 8 — Evidence & limitations */}
      <LimitationsSection limitations={report.limitations} />

      <footer className="dash-footer mono">
        {source === 'static' ? (
          <span>Data source · bundled report copy</span>
        ) : (
          <span>Data source · API (GET /optimization/report)</span>
        )}
        {report.meta.generatedAt && (
          <span>Report generated · {new Date(report.meta.generatedAt).toLocaleString()}</span>
        )}
        {report.meta.regenerateCommand && (
          <span>Regenerate · {report.meta.regenerateCommand}</span>
        )}
      </footer>
    </div>
  )
}
