import React from 'react';
import './TypingIndicator.css';

/**
 * WhatsApp-style typing indicator with three bouncing dots
 */
export default function TypingIndicator() {
  return (
    <div className="message bot typing-indicator-container">
      <div className="typing-indicator">
        <span className="dot"></span>
        <span className="dot"></span>
        <span className="dot"></span>
      </div>
    </div>
  );
}
