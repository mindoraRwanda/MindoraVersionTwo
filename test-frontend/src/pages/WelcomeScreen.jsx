import React, { useState, useEffect } from 'react';

// WelcomeScreen Component for displaying the initial welcome interface
const WelcomeScreen = ({ onSuggestionClick }) => {
  const [showPrompt, setShowPrompt] = useState(false);
  const [visibleSuggestions, setVisibleSuggestions] = useState([]);

  const suggestions = [
    { icon: '🌤️', text: 'talk about your day' },
    { icon: '💪', text: 'work through something challenging' },
    { icon: '💬', text: 'just need someone to chat with' }
  ];

  useEffect(() => {
    const promptTimer = setTimeout(() => {
      setShowPrompt(true);
    }, 1000);

    // Animate suggestions one by one
    const timers = [];
    suggestions.forEach((_, index) => {
      const timer = setTimeout(() => {
        setVisibleSuggestions(prev => {
          if (!prev.includes(index)) {
            return [...prev, index];
          }
          return prev;
        });
      }, 1400 + (index * 500));
      timers.push(timer);
    });

    return () => {
      clearTimeout(promptTimer);
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [suggestions]);

  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="welcome-icon-container">
          <div className="welcome-icon">💜</div>
          <div className="ripple ripple-1"></div>
          <div className="ripple ripple-2"></div>
          <div className="ripple ripple-3"></div>
        </div>
        
        <h1 className="welcome-title">
          Hey there, welcome!
        </h1>
        
        <p className="welcome-description">
          I'm here to listen, understand, and support you through whatever's on your mind.
        </p>

        <div className="suggestions-container">
          <p className="suggestions-intro">Whether you want to:</p>
          <div className="suggestions-list">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                className={`suggestion-item ${visibleSuggestions.includes(index) ? 'suggestion-visible' : ''}`}
                style={{ transitionDelay: `${index * 0.1}s` }}
                onClick={() => onSuggestionClick && onSuggestionClick(`I'd like to ${suggestion.text}`)}
                type="button"
              >
                <span className="suggestion-icon">{suggestion.icon}</span>
                <span className="suggestion-text">{suggestion.text}</span>
              </button>
            ))}
          </div>
          <p className="suggestions-outro">—I've got you.</p>
        </div>
        
        {showPrompt && (
          <p className="welcome-prompt">
            How are you feeling today
            <span className="dot-flashing-container">
              <span className="dot-flashing">.</span>
              <span className="dot-flashing">.</span>
              <span className="dot-flashing">.</span>
            </span>
          </p>
        )}

        <div className="floating-particles">
          <span className="particle particle-1">✨</span>
          <span className="particle particle-2">💫</span>
          <span className="particle particle-3">⭐</span>
          <span className="particle particle-4">🌟</span>
          <span className="particle particle-5">✨</span>
          <span className="particle particle-6">💫</span>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;