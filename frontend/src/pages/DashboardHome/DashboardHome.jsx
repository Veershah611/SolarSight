import { DiagnosticsPage } from '../DiagnosticsPage/DiagnosticsPage';
import { RiskOverviewTable } from '../../components/RiskOverviewTable/RiskOverviewTable';
import { LayoutGrid } from 'lucide-react';
import styles from './DashboardHome.module.css';

function FleetSummary({ summary }) {
  if (!summary) return null;
  const ok = summary.inverters - summary.inverters_in_alarm - summary.inverters_on_watch;
  const tiles = [
    { label: 'In alarm', value: summary.inverters_in_alarm, color: 'var(--status-high)' },
    { label: 'On watch', value: summary.inverters_on_watch, color: 'var(--status-medium)' },
    { label: 'Normal', value: ok, color: 'var(--status-low)' },
    { label: 'Latest day', value: summary.latest_day, color: 'var(--text-heading)' },
  ];
  return (
    <div className={styles.summary}>
      {tiles.map((t) => (
        <div key={t.label} className={styles.summaryTile}>
          <div className={styles.summaryValue} style={{ color: t.color }}>{t.value}</div>
          <div className={styles.summaryLabel}>{t.label}</div>
        </div>
      ))}
      <p className={styles.summaryNote}>
        {summary.inverters} inverters · {summary.period_start.slice(0, 10)} to {summary.period_end.slice(0, 10)} ·
        {' '}{summary.judged_hours.toLocaleString()} inverter-hours judged against their peers
      </p>
    </div>
  );
}

export function DashboardHome({ inverters, summary, selectedInverter, onSelectInverter }) {
  return (
    <div className={styles.dashboardHome}>
      <div className={styles.sectionHeader}>
        <div style={{ textAlign: "center", width: "100%" }}>
          <h1 className={styles.sectionTitle}>
            {selectedInverter ? `Diagnostics: ${selectedInverter.inverter_id}` : "Fleet Overview"}
          </h1>
          <p className={styles.sectionSubtitle}>
            {selectedInverter
              ? "Actual output against what its peers say it should be producing"
              : "Inverters producing less than their peers imply, detected hour by hour"}
          </p>
        </div>
        {selectedInverter && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginTop: "12px" }}>
            <button className={styles.backButton} onClick={() => onSelectInverter(null)}>
              <LayoutGrid size={16} />
              Back to Fleet
            </button>
          </div>
        )}
      </div>

      {selectedInverter ? (
        <DiagnosticsPage inverter={selectedInverter} />
      ) : inverters.length > 0 ? (
        <div style={{ animation: "fadeIn 0.5s ease-out" }}>
          <FleetSummary summary={summary} />
          <RiskOverviewTable inverters={inverters} onSelectInverter={onSelectInverter} />
        </div>
      ) : (
        <div style={{ textAlign: "center", padding: "80px 20px", color: "#666" }}>
          <div style={{ fontSize: "40px", marginBottom: "16px" }}>📊</div>
          <div style={{ fontSize: "16px", marginBottom: "8px" }}>No Data Loaded</div>
          <div style={{ fontSize: "14px" }}>Run the Analyzer on a data-logger export to see each inverter's status.</div>
        </div>
      )}
    </div>
  );
}
