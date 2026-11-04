import { useMemo, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Bot, Sparkles } from 'lucide-react';
import { api } from '../../services/api';
import { EVIDENCE_LABELS, RISK_LEVELS, STATUS_TEXT } from '../../utils/constants';
import styles from './DiagnosticsPage.module.css';

const CATEGORY_COLORS = {
  electrical: '#4cc9f0', string: '#ffd60a', thermal: '#ff6b35',
  soiling_shading: '#06d6a0', communication: '#c77dff', other: '#888',
};

const tooltipProps = {
  contentStyle: { background: 'rgba(10, 10, 15, 0.9)', border: '1px solid #1e1e2e', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#4cc9f0', marginBottom: 4 },
};

function formatEvidence(signal, value) {
  if (value == null) return '—';
  switch (EVIDENCE_LABELS[signal]?.format) {
    case 'pct': return `${Math.round(value * 100)}%`;
    case 'int': return String(Math.round(value));
    case 'celsius': return `${value > 0 ? '+' : ''}${value.toFixed(1)} °C`;
    default: return value.toFixed(2);
  }
}

function Kpi({ label, value, color }) {
  return (
    <div className={styles.card}>
      <div className={styles.kpiValue} style={{ color }}>{value}</div>
      <div className={styles.kpiLabel}>{label}</div>
    </div>
  );
}

function OutputChart({ hourly }) {
  const data = useMemo(() => hourly.map((h) => ({
    ...h,
    alarm_point: h.alarm ? h.actual_pu : null,
  })), [hourly]);
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" vertical={false} />
        <XAxis dataKey="timestamp" tick={{ fill: '#666', fontSize: 10 }} minTickGap={40}
          tickFormatter={(t) => t.slice(5, 13).replace(' ', ' ')} />
        <YAxis tick={{ fill: '#666', fontSize: 10 }} domain={[0, 'auto']}
          tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip {...tooltipProps} formatter={(v, name) => [v == null ? '—' : `${Math.round(v * 100)}% of capacity`, name]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="expected_pu" name="Expected (from peers)" stroke="#888"
          strokeDasharray="4 3" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="actual_pu" name="Actual" stroke="#4cc9f0" dot={false} strokeWidth={2} />
        {/* A line with no stroke: only hours with an alarm (non-null) get a dot */}
        <Line dataKey="alarm_point" name="Alarm" stroke="none" legendType="circle" isAnimationActive={false}
          dot={{ r: 3, fill: '#ff2d55', stroke: '#ff2d55' }} activeDot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function DailyChart({ daily }) {
  const data = daily.map((d) => ({ ...d, pct: d.energy_ratio == null ? null : Math.round(d.energy_ratio * 100) }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" vertical={false} />
        <XAxis dataKey="day" tick={{ fill: '#666', fontSize: 10 }} tickFormatter={(d) => d.slice(5)} />
        <YAxis tick={{ fill: '#666', fontSize: 10 }} domain={[0, 120]} tickFormatter={(v) => `${v}%`} />
        <Tooltip {...tooltipProps} formatter={(v) => [`${v}% of expected`, 'Output']} />
        <ReferenceLine y={100} stroke="#555" strokeDasharray="3 3" />
        <Bar dataKey="pct" radius={[2, 2, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.day} fill={d.alarm ? '#ff2d55' : d.pct != null && d.pct < 90 ? '#ff9500' : '#06d6a0'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function Report({ result }) {
  const { report, source } = result;
  // Rule-based reports repeat each cause's action as a check; only list checks that add something
  const actions = new Set(report.likelyCauses?.map((c) => c.action));
  const checks = report.checks?.filter((c) => !actions.has(c)) ?? [];
  return (
    <div className={styles.report}>
      <div className={styles.reportHeader}>
        <div>
          <div className={styles.reportHeadline}>{report.headline}</div>
          <div className={styles.reportSummary}>{report.summary}</div>
        </div>
        <span className={styles.sourceBadge}>{source === 'gemini' ? 'Gemini' : 'Rule-based'}</span>
      </div>
      {report.likelyCauses?.map((c) => (
        <div key={c.rank} className={styles.cause} style={{ borderLeftColor: CATEGORY_COLORS[c.category] || '#4cc9f0' }}>
          <div className={styles.causeTop}>
            <span className={styles.causeName}>{c.rank}. {c.name}</span>
            <span className={styles.causeConfidence}>{c.confidence}% confidence</span>
          </div>
          <div className={styles.causeEvidence}>Evidence: {c.evidence}</div>
          <div className={styles.causeAction}>→ {c.action}</div>
        </div>
      ))}
      {checks.length > 0 && (
        <ol className={styles.checks}>
          {checks.map((c, i) => <li key={i}>{c}</li>)}
        </ol>
      )}
    </div>
  );
}

export function DiagnosticsPage({ inverter }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const level = RISK_LEVELS[inverter.risk_label];
  const pct = (r) => (r == null ? '—' : `${Math.round(r * 100)}%`);

  async function diagnose() {
    setLoading(true);
    setError(null);
    try {
      setReport(await api.explain(inverter));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.kpis}>
        <Kpi label="Status" value={STATUS_TEXT[inverter.status]} color={level?.color} />
        <Kpi label="Output vs expected, latest day" value={pct(inverter.energy_ratio_latest_day)} />
        <Kpi label="Output vs expected, whole upload" value={pct(inverter.energy_ratio_window)} />
        <Kpi label="Alarm hours" value={inverter.alarm_hours}
          color={inverter.alarm_hours ? 'var(--status-high)' : undefined} />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Hourly output, last 7 days (share of capacity)</div>
        <OutputChart hourly={inverter.hourly} />
        <p className={styles.caption}>
          Expected output comes from an XGBoost model fed only with the other inverters' output, so this
          inverter's own fault can't hide itself. Red points mark hours where the CUSUM alarm fired.
          {inverter.first_alarm && ` First alarm: ${inverter.first_alarm.slice(0, 16)}.`}
        </p>
      </div>

      <div className={styles.grid}>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Daily output vs expected</div>
          <DailyChart daily={inverter.daily} />
        </div>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Evidence (latest day vs typical)</div>
          <div className={styles.evidence}>
            {inverter.evidence.map((e) => (
              <div key={e.signal} className={styles.evidenceItem} title={e.description}>
                <div className={styles.evidenceLabel}>{EVIDENCE_LABELS[e.signal]?.label || e.signal}</div>
                <div className={styles.evidenceValue}>{formatEvidence(e.signal, e.value)}</div>
                {e.typical != null && (
                  <div className={styles.evidenceTypical}>typical {formatEvidence(e.signal, e.typical)}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}><Bot size={16} /> Maintenance diagnosis</div>
        {!report && (
          <div className={styles.diagnosePrompt}>
            <p>Turn the evidence above into likely causes and on-site checks. Uses Gemini when an API key is
              configured, otherwise transparent rules.</p>
            <button className={styles.diagnoseButton} onClick={diagnose} disabled={loading}>
              <Sparkles size={14} /> {loading ? 'Diagnosing…' : 'Diagnose'}
            </button>
            {error && <div className={styles.error}>{error}</div>}
          </div>
        )}
        {report && <Report result={report} />}
      </div>
    </div>
  );
}
