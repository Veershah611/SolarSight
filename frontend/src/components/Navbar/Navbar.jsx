import {
  BarChart2,
  LayoutDashboard,
  AlertTriangle,
  TrendingUp,
  MessageSquare,
  Beaker,
  Sun
} from 'lucide-react';
import styles from './Navbar.module.css';
const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'qa', label: 'Operator Q&A', icon: MessageSquare },
  { id: 'sandbox', label: 'Analyzer', icon: Beaker },

];

export function Navbar({ activePage, onPageChange, apiHealthy }) {
  return (
    <nav className={styles.navbar}>
      <div className={styles.navLeft}>
        <div className={styles.brand}>
          <Sun className={styles.brandIcon} size={24} />
          <span className={styles.brandText}>SolarSight</span>
        </div>

        <div className={styles.tabs}>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`${styles.tabBtn} ${activePage === item.id ? styles.active : ''}`}
                onClick={() => onPageChange(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className={styles.navRight}>
        {apiHealthy !== null && (
          <div className={`${styles.statusBadge} ${apiHealthy ? styles.statusOk : styles.statusOff}`}>
            <span className={styles.statusDot}></span>
            {apiHealthy ? 'Connected' : 'Offline'}
          </div>
        )}
      </div>
    </nav>
  );
}
