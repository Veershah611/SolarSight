import { useState, useRef, useEffect } from 'react';
import { api } from '../../services/api';
import { Beaker, UploadCloud, FileSpreadsheet } from 'lucide-react';
import styles from './SandboxPage.module.css';

// One week of a real 12-inverter logger export (plant 1), ending on a day one inverter lost most of its output
const DEMO_FILE = { path: '/ICR2-LT1.raws.csv', name: 'ICR2-LT1.raws.csv' };

export function SandboxPage({ onResultsUpdate }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Preload the demo export so the analyzer works in one click
  useEffect(() => {
    async function loadDemo() {
      try {
        const response = await fetch(DEMO_FILE.path);
        if (!response.ok) return;
        const blob = await response.blob();
        setFiles((current) => current.length ? current : [new File([blob], DEMO_FILE.name, { type: 'text/csv' })]);
      } catch {
        // Silently ignore – user can still upload manually
      }
    }
    loadDemo();
  }, []);

  const selectFiles = (list) => {
    const csvs = Array.from(list).filter((f) => f.name.toLowerCase().endsWith('.csv'));
    if (csvs.length === 0) {
      setError('Please choose one or more .csv data-logger exports.');
      return;
    }
    setFiles(csvs);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files?.length) selectFiles(e.dataTransfer.files);
  };

  const handleDetect = async () => {
    if (files.length === 0) {
      setError('Please select a CSV file first.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.detect(files);
      onResultsUpdate?.(data);
    } catch (err) {
      setError(err.message || 'Detection failed.');
    } finally {
      setLoading(false);
    }
  };

  const totalKb = files.reduce((sum, f) => sum + f.size, 0) / 1024;

  return (
    <div className={styles.sandboxPage}>
      <header className={styles.header}>
        <div className={styles.headerTitleWrapper}>
          <Beaker className={styles.headerIcon} size={28} />
          <h1 className={styles.title}>Underperformance Analyzer</h1>
        </div>
        <p className={styles.description}>
          Upload raw data-logger exports from one plant (several days, ideally a week or more).
          Each inverter is compared hour by hour with what its peers say it should be producing.
        </p>
      </header>

      <div className={styles.content}>
        <div className={styles.card}>
          <div
            className={styles.dropZone}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              accept=".csv"
              multiple
              ref={fileInputRef}
              onChange={(e) => e.target.files?.length && selectFiles(e.target.files)}
              style={{ display: 'none' }}
            />
            {files.length ? (
              <div className={styles.fileSelected}>
                <FileSpreadsheet className={styles.fileIcon} size={32} />
                {files.map((f) => <span key={f.name} className={styles.fileName}>{f.name}</span>)}
                <p className={styles.fileSize}>{totalKb.toFixed(0)} KB · click to choose different files</p>
              </div>
            ) : (
              <div className={styles.dropPrompt}>
                <UploadCloud className={styles.uploadIcon} size={48} />
                <p>Click or drag and drop logger exports (.csv) here</p>
              </div>
            )}
          </div>

          <button
            className={styles.predictBtn}
            onClick={handleDetect}
            disabled={loading || files.length === 0}
          >
            {loading ? 'Comparing inverters with their peers…' : 'Run Detection'}
          </button>

          {error && <div className={styles.error}>{error}</div>}
        </div>
      </div>
    </div>
  );
}
