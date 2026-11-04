import { OperatorQA } from '../../components/OperatorQA/OperatorQA';
import { MessageSquare } from 'lucide-react';
import styles from './QAPage.module.css';

export function QAPage({ onQuery, qaResponses }) {
  return (
    <div className={styles.qaPage}>
      <div className={styles.pageHeader}>
        <div className={styles.headerTitleWrapper}>
          <MessageSquare className={styles.headerIcon} size={28} />
          <h1 className={styles.pageTitle}>Operator Assistant</h1>
        </div>
        <p className={styles.pageDescription}>
          Ask questions about inverter performance, risks, and maintenance
        </p>
      </div>

      <div className={styles.qaContainer}>
        <OperatorQA onQuery={onQuery} responses={qaResponses} />
      </div>

      <div className={styles.suggestions}>
        <h3 className={styles.suggestionsTitle}>Suggested Questions:</h3>
        <div className={styles.suggestionChips}>
          {/* Static General Questions */}
          <button className={styles.chip} onClick={() => onQuery("Which inverters are in alarm, and how much output are they losing?")}>
            Which inverters are in alarm?
          </button>
          <button className={styles.chip} onClick={() => onQuery("What should the technician check first on the inverters in alarm?")}>
            What should we check on site?
          </button>

        </div>
      </div>
    </div>
  );
}
