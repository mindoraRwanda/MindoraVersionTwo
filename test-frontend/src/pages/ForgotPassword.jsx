import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { forgotPassword } from '../api/api';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!email) {
      setStatus('Please enter your email');
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      await forgotPassword(email.trim());
    } catch (error) {
      // Backend always returns a generic success message regardless of
      // whether the email exists, so a request error here is a real
      // network/server problem, not "email not found".
      console.error('Forgot password error:', error.response?.data || error.message);
    } finally {
      setLoading(false);
      // Always show the same confirmation, whether or not the email was
      // actually registered — this endpoint intentionally doesn't reveal that.
      setSubmitted(true);
    }
  };

  return (
    <div className="login-container">
      <style>{`
        body {
          margin: 0;
          font-family: sans-serif;
          background-color: #f3f4f6;
        }
        .login-container {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .login-box {
          max-width: 440px;
          width: 90%;
          background-color: white;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
          border-radius: 10px;
          overflow: hidden;
        }
        .login-form {
          padding: 40px;
          background-color: #fff;
        }
        .brand {
          color: rgb(109, 40, 217);
          font-size: 32px;
          margin-bottom: 10px;
        }
        .form-title {
          font-size: 24px;
          font-weight: bold;
          color: #111827;
          margin-bottom: 12px;
        }
        .form-subtitle {
          color: #6b7280;
          font-size: 14px;
          margin-bottom: 20px;
          line-height: 1.5;
        }
        .input-field {
          width: 100%;
          padding: 12px 16px;
          margin-bottom: 16px;
          border: 1px solid #ccc;
          border-radius: 25px;
          background-color: #f9fafb;
          transition: border-color 0.3s ease;
          box-sizing: border-box;
          font-size: 16px;
        }
        .input-field:focus {
          outline: none;
          border-color: rgb(109, 40, 217);
        }
        .submit-btn {
          width: 100%;
          padding: 12px;
          background: linear-gradient(to right, rgb(109, 40, 217), rgb(91, 33, 182));
          color: white;
          font-weight: bold;
          border: none;
          border-radius: 25px;
          cursor: pointer;
          transition: background 0.3s ease;
          box-sizing: border-box;
          font-size: 15px;
        }
        .submit-btn:hover {
          background: linear-gradient(to right, rgb(91, 33, 182), rgb(79, 70, 229));
        }
        .submit-btn:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }
        .link-btn {
          background: none;
          border: none;
          color: #2563eb;
          cursor: pointer;
          text-decoration: none;
          font-size: 14px;
          margin-top: 16px;
        }
        .error-msg {
          margin-top: 16px;
          color: #dc2626;
          font-size: 14px;
        }
        .success-msg {
          color: #334155;
          font-size: 15px;
          line-height: 1.6;
        }

        @media (max-width: 400px) {
          .login-form {
            padding: 28px 22px;
          }
          .brand {
            font-size: 26px;
          }
        }
      `}</style>

      <div className="login-box">
        <div className="login-form">
          <h1 className="brand">Mindora Bot</h1>

          {submitted ? (
            <>
              <h2 className="form-title">Check your email</h2>
              <p className="success-msg">
                If that email is registered, we've sent a link to reset your password.
                It expires in 30 minutes.
              </p>
              <button onClick={() => navigate('/')} className="link-btn">
                Back to Sign In
              </button>
            </>
          ) : (
            <>
              <h2 className="form-title">Forgot password?</h2>
              <p className="form-subtitle">
                Enter the email on your account and we'll send you a link to reset your password.
              </p>

              <form onSubmit={handleSubmit}>
                <input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="input-field"
                />

                <button type="submit" className="submit-btn" disabled={loading}>
                  {loading ? 'Sending...' : 'Send Reset Link'}
                </button>
              </form>

              <button onClick={() => navigate('/')} className="link-btn">
                Back to Sign In
              </button>

              {status && <p className="error-msg">{status}</p>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
