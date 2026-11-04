import { AlertTriangle } from 'lucide-react';
import { RISK_LEVELS, STATUS_TEXT } from '../../utils/constants.js';
import styles from './RiskOverviewTable.module.css';

const pct = (ratio) => (ratio == null ? '—' : `${Math.round(ratio * 100)}%`);

/**
 * Fleet table — inverters ordered by detector status (ALARM, WATCH, OK) then shortfall.
 */
export function RiskOverviewTable({ inverters = [], onSelectInverter }) {
  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Inverter Status</h2>
      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th></th>
              <th>Inverter</th>
              <th>Status</th>
              <th>Output vs expected (latest day)</th>
              <th>Output vs expected (whole upload)</th>
              <th>Alarm hours</th>
              <th>First alarm</th>
            </tr>
          </thead>
          <tbody>
            {inverters.length === 0 ? (
              <tr>
                <td colSpan={7} className={styles.empty}>
                  No results yet. Upload a logger export in the Analyzer tab.
                </td>
              </tr>
            ) : (
              inverters.map((inv) => {
                const level = RISK_LEVELS[inv.risk_label];
                const isAlarm = inv.status === 'ALARM';
                return (
                  <tr
                    key={inv.inverter_id}
                    onClick={() => onSelectInverter?.(inv)}
                    className={`${styles.row} ${isAlarm ? styles.highRiskRow : ''}`}
                  >
                    <td className={styles.statusCell}>
                      {isAlarm ? (
                        <AlertTriangle className={styles.dangerIcon} size={20} />
                      ) : (
                        <div className={styles.statusDot} style={{ backgroundColor: level?.color }} />
                      )}
                    </td>
                    <td className={styles.idCell}>{inv.inverter_id}</td>
                    <td>
                      <span className={styles.badge} style={{ color: level?.color, backgroundColor: level?.bgColor }}>
                        {STATUS_TEXT[inv.status]}
                      </span>
                    </td>
                    <td className={styles.scoreCell}>
                      <span className={styles.scoreValue}>{pct(inv.energy_ratio_latest_day)}</span>
                    </td>
                    <td className={styles.scoreCell}>{pct(inv.energy_ratio_window)}</td>
                    <td>{inv.alarm_hours}</td>
                    <td className={styles.windowCell}>{inv.first_alarm ? inv.first_alarm.slice(0, 16) : '—'}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
