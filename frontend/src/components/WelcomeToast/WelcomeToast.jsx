import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Info, X } from 'lucide-react';
import styles from './WelcomeToast.module.css';

export const WelcomeToast = ({ show, onClose }) => {
    useEffect(() => {
        if (show) {
            const timer = setTimeout(() => {
                onClose();
            }, 8000); // Show for 8 seconds
            return () => clearTimeout(timer);
        }
    }, [show, onClose]);

    return (
        <AnimatePresence>
            {show && (
                <motion.div
                    className={styles.toastContainer}
                    initial={{ y: -100, opacity: 0, x: '-50%' }}
                    animate={{ y: 30, opacity: 1, x: '-50%' }}
                    exit={{ y: -100, opacity: 0, x: '-50%' }}
                    transition={{ type: 'spring', stiffness: 260, damping: 20 }}
                >
                    <div className={styles.toast}>
                        <div className={styles.iconContainer}>
                            <div className={styles.pulseBubble} />
                            <Info size={24} className={styles.infoIcon} />
                        </div>
                        <div className={styles.content}>
                            <h4 className={styles.title}>Welcome to SolarSight!</h4>
                            <p className={styles.message}>
                                Open the <strong>Analyzer</strong> tab and run the bundled demo export (or upload your own) to see which inverters are underperforming.
                            </p>
                        </div>
                        <button className={styles.closeBtn} onClick={onClose} aria-label="Close message">
                            <X size={18} />
                        </button>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
