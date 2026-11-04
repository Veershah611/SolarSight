/**
 * API service for SolarSight FastAPI backend
 * Endpoints: /health, /detect, /explain, /query
 */

// Endpoints start with "/", so drop any trailing slash from the configured URL (avoids "//detect" 404s)
const API_BASE = (import.meta.env.VITE_API_URL || '/api').replace(/\/+$/, '');

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };
  const response = await fetch(url, config);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/** The fields of a detection result the GenAI endpoints use (drops the chart series). */
export function toContext(inv) {
  if (!inv) return null;
  return {
    inverter_id: inv.inverter_id,
    status: inv.status,
    risk_score: inv.risk_score,
    energy_ratio_latest_day: inv.energy_ratio_latest_day,
    energy_ratio_window: inv.energy_ratio_window,
    alarm_hours: inv.alarm_hours,
    first_alarm: inv.first_alarm,
    evidence: inv.evidence,
  };
}

export const api = {
  /** GET /health — Health check */
  async getHealth() {
    return request('/health');
  },

  /** POST /explain — maintenance report for one inverter (Gemini or rule-based) */
  async explain(inverter, parameters = null) {
    return request('/explain', {
      method: 'POST',
      body: JSON.stringify({ inverter: toContext(inverter), parameters }),
    });
  },

  /** POST /query — operator Q&A over the latest detection results */
  async query(question, inverter = null, fleet = [], parameters = null) {
    return request('/query', {
      method: 'POST',
      body: JSON.stringify({
        question,
        inverter: toContext(inverter),
        fleet: fleet.map(toContext),
        parameters,
      }),
    });
  },

  /** POST /detect — score raw data-logger exports from one plant */
  async detect(files) {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    // Not using request(): the browser must set the multipart boundary itself
    const response = await fetch(`${API_BASE}/detect`, {
      method: 'POST',
      body: formData,
    });

    if (response.status === 404) {
      throw new Error('The backend has no /detect endpoint. It is running an older version of the API; redeploy it from main.');
    }
    if (!response.ok) {
      let errorText = await response.text();
      try {
        const errObj = JSON.parse(errorText);
        errorText = errObj.detail || errorText;
      } catch (e) { }
      throw new Error(errorText);
    }
    return response.json();
  },
};
