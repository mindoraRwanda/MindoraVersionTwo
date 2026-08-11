import React, { useEffect, useState } from 'react';
import './CrisisHelpModal.css';

const RESOURCES = [
  {
    label: 'Rwanda Mental Health Helpline',
    detail: 'Free, confidential, 24/7',
    icon: '📞',
    value: '114',
  },
  {
    label: 'Emergency Services / Police',
    detail: 'Immediate danger, right now',
    icon: '📞',
    value: '112',
  },
  {
    label: 'Mindora Support Line',
    detail: 'Talk to our team directly',
    icon: '📞',
    value: '+250783974066',
    display: '+250 783 974 066',
  },
  {
    label: 'Mindora Support (Email)',
    detail: "We'll respond as soon as we can",
    icon: '📧',
    value: 'Info@mindora.rw',
  },
];

// Falls back to a hidden-textarea + execCommand for browsers/contexts where
// the async Clipboard API isn't available (e.g. non-HTTPS, older browsers).
function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand('copy');
  } finally {
    document.body.removeChild(textarea);
  }
  return Promise.resolve();
}

export default function CrisisHelpModal({ onClose }) {
  const [copiedLabel, setCopiedLabel] = useState(null);

  // Close on Escape key
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  const handleCopy = async (resource) => {
    try {
      await copyToClipboard(resource.display || resource.value);
      setCopiedLabel(resource.label);
      setTimeout(() => setCopiedLabel((prev) => (prev === resource.label ? null : prev)), 1500);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };

  return (
    <div className="chm-overlay" onClick={onClose}>
      <div className="chm-card" onClick={(e) => e.stopPropagation()}>
        <div className="chm-icon">🆘</div>

        <h2 className="chm-title">You don't have to handle this alone</h2>
        <p className="chm-body">
          If you're in immediate danger or need to talk to someone right now, these are
          real people trained to help — free, confidential, and available right now.
        </p>

        <div className="chm-resources">
          {RESOURCES.map((r) => (
            <button
              key={r.label}
              type="button"
              className="chm-resource"
              onClick={() => handleCopy(r)}
            >
              <div className="chm-resource-text">
                <span className="chm-resource-label">{r.label}</span>
                <span className="chm-resource-detail">{r.detail}</span>
              </div>
              <span className="chm-resource-phone">
                {copiedLabel === r.label ? '✅ Copied!' : `${r.icon} ${r.display || r.value}`}
              </span>
            </button>
          ))}
        </div>

        <button className="chm-close" onClick={onClose} autoFocus>
          Close
        </button>
      </div>
    </div>
  );
}
