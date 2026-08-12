import { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { resetPassword } from '../api/api';

export default function ResetPassword() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token') || '';

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [status, setStatus] = useState(null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!token) {
      setStatus('This reset link is missing its token — please request a new one.');
      return;
    }
    if (!password || !confirmPassword) {
      setStatus('Please fill in both password fields');
      return;
    }
    if (password.length < 6) {
      setStatus('Password must be at least 6 characters');
      return;
    }
    if (password !== confirmPassword) {
      setStatus('Passwords do not match');
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      await resetPassword(token, password);
      setSuccess(true);
    } catch (error) {
      const detail = error.response?.data?.detail;
      setStatus(
        typeof detail === 'string'
          ? detail
          : 'That reset link is invalid or has expired. Please request a new one.'
      );
    } finally {
      setLoading(false);
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
          margin-bottom: 20px;
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

          {success ? (
            <>
              <h2 className="form-title">Password reset</h2>
              <p className="success-msg">
                Your password has been updated. You can now sign in with your new password.
              </p>
              <button onClick={() => navigate('/')} className="submit-btn" style={{ marginTop: '16px' }}>
                Go to Sign In
              </button>
            </>
          ) : (
            <>
              <h2 className="form-title">Set a new password</h2>

              <form onSubmit={handleSubmit}>
                <input
                  type="password"
                  placeholder="New password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="input-field"
                />
                <input
                  type="password"
                  placeholder="Confirm new password"
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  className="input-field"
                />

                <button type="submit" className="submit-btn" disabled={loading}>
                  {loading ? 'Resetting...' : 'Reset Password'}
                </button>
              </form>

              <button onClick={() => navigate('/forgot-password')} className="link-btn">
                Request a new link
              </button>

              {status && <p className="error-msg">{status}</p>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
