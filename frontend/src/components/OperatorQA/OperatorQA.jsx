import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import styles from './OperatorQA.module.css';

/**
 * Operator Q&A — Chat interface for natural language queries
 * about the latest detection results (POST /query)
 */
export function OperatorQA({ onQuery, responses = [] }) {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    setIsLoading(true);
    try {
      await onQuery?.(question.trim());
      setQuestion('');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Operator Q&A</h2>
      <p className={styles.subtitle}>
        Ask about underperforming inverters and what to check on site
      </p>

      <div className={styles.chatArea}>
        {responses.length === 0 && !isLoading && (
          <div className={styles.placeholder}>
            e.g. &quot;Which inverters are in alarm, and what should the technician check first?&quot;
          </div>
        )}
        {responses.map((item, i) => (
          <div key={i} className={styles.exchange}>
            <div className={styles.question}>
              <strong>Q:</strong> {item.question}
            </div>
            <div className={styles.answer}>
              <strong>A:</strong>
              <div style={{ marginTop: "8px", lineHeight: "1.6", fontSize: "14px" }}>
                <ReactMarkdown>{item.answer}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className={styles.loading}>Searching...</div>
        )}
      </div>

      <form onSubmit={handleSubmit} className={styles.form}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
          className={styles.input}
          disabled={isLoading}
        />
        <button
          type="submit"
          className={styles.button}
          disabled={isLoading || !question.trim()}
        >
          Ask
        </button>
      </form>
    </div>
  );
}
