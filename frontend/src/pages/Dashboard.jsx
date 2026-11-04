import { useState, useCallback, useEffect } from 'react';
import { Navbar } from '../components/Navbar/Navbar';
import { DashboardHome } from './DashboardHome/DashboardHome';
import { QAPage } from './QAPage/QAPage';
import { SandboxPage } from './SandboxPage/SandboxPage';
import { api } from '../services/api';
import styles from './Dashboard.module.css';

export function Dashboard() {
  const [activePage, setActivePage] = useState('dashboard');

  const [inverters, setInverters] = useState([]);
  const [summary, setSummary] = useState(null);
  const [selectedInverter, setSelectedInverter] = useState(null);
  const [qaResponses, setQAResponses] = useState([]);
  const [apiHealthy, setApiHealthy] = useState(null);

  useEffect(() => {
    api.getHealth()
      .then(() => setApiHealthy(true))
      .catch(() => setApiHealthy(false));
  }, []);

  const handleQuery = useCallback(async (question) => {
    try {
      const res = await api.query(question, selectedInverter, inverters);
      setQAResponses((prev) => [...prev, { question, answer: res.answer }]);
    } catch (err) {
      setQAResponses((prev) => [
        ...prev,
        { question, answer: `Error: ${err.message}. (API may be unavailable.)` },
      ]);
    }
  }, [selectedInverter, inverters]);

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return (
          <DashboardHome
            inverters={inverters}
            summary={summary}
            selectedInverter={selectedInverter}
            onSelectInverter={setSelectedInverter}
          />
        );
      case 'qa':
        return <QAPage onQuery={handleQuery} qaResponses={qaResponses} />;
      case 'sandbox':
        return (
          <SandboxPage
            onResultsUpdate={(data) => {
              setInverters(data.results);
              setSummary(data.summary);
              setSelectedInverter(null);
              setActivePage('dashboard');
            }}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className={styles.dashboard}>
      <Navbar
        activePage={activePage}
        onPageChange={setActivePage}
        apiHealthy={apiHealthy}
      />
      <main className={styles.main}>
        <div className={styles.content}>
          {renderPage()}
        </div>
      </main>
    </div>
  );
}
