/** Severity levels and their styling (the API maps ALARM/WATCH/OK to HIGH/MEDIUM/LOW) */
export const RISK_LEVELS = {
  LOW: { label: 'Low', color: '#22C55E', bgColor: 'rgba(34, 197, 94, 0.15)' },
  MEDIUM: { label: 'Medium', color: '#F59E0B', bgColor: 'rgba(245, 158, 11, 0.15)' },
  HIGH: { label: 'High', color: '#EF4444', bgColor: 'rgba(239, 68, 68, 0.15)' },
};

export const STATUS_TEXT = { ALARM: 'Alarm', WATCH: 'Watch', OK: 'OK' };

export const STATUS_RISK = { ALARM: 'HIGH', WATCH: 'MEDIUM', OK: 'LOW' };

/** Human-readable names for the detector's evidence signals */
export const EVIDENCE_LABELS = {
  energy_vs_expected: { label: 'Output vs expected', format: 'pct' },
  zero_output_hours: { label: 'Zero-output daylight hours', format: 'int' },
  output_vs_peers: { label: 'Output / peer median', format: 'ratio' },
  temp_vs_peers_c: { label: 'Temperature vs peers', format: 'celsius' },
  string_balance: { label: 'Weakest / strongest string', format: 'ratio' },
};
