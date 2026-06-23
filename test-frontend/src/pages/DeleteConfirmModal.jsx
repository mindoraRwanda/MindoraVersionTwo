import React, { useEffect } from 'react';
import './DeleteConfirmModal.css';

export default function DeleteConfirmModal({ chatLabel, onConfirm, onCancel, isDeleting }) {
  // Close on Escape key
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onCancel]);

  return (
    <div className="dcm-overlay" onClick={onCancel}>
      <div className="dcm-card" onClick={(e) => e.stopPropagation()}>
        <div className="dcm-icon">🗑️</div>

        <h2 className="dcm-title">Delete conversation?</h2>
        <p className="dcm-body">
          <span className="dcm-label">"{chatLabel}"</span> and all its messages will be
          permanently deleted. This cannot be undone.
        </p>

        <div className="dcm-actions">
          <button className="dcm-cancel" onClick={onCancel} disabled={isDeleting}>
            Cancel
          </button>
          <button className="dcm-confirm" onClick={onConfirm} disabled={isDeleting} autoFocus>
            {isDeleting ? 'Deleting…' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
}
