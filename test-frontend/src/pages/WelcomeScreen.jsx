// WelcomeScreen Component for displaying the initial welcome interface
const WelcomeScreen = () => {
  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <h1 className="welcome-title">
          Welcome to MINDORA
        </h1>
        <p className="welcome-description">
          Your AI-powered mental health companion for Rwandan youth.
        </p>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <div className="feature-title">Emotion Detection</div>
            <div className="feature-description">Understands your emotional state</div>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🌍</div>
            <div className="feature-title">Cultural Context</div>
            <div className="feature-description">Designed for Rwandan values</div>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <div className="feature-title">Safe & Private</div>
            <div className="feature-description">Secure conversations</div>
          </div>
        </div>
        <p className="welcome-footer">
          Start typing below to begin our conversation.
        </p>
      </div>
    </div>
  );
};

export default WelcomeScreen;