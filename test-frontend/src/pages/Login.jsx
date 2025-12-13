import { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async () => {
  if (!email || !password) {
    setStatus('Please enter both email and password');
    return;
  }

  console.log('Sending login request:', { email, password: '***' });
  setLoading(true);

  try {
    const res = await axios.post('http://localhost:8000/auth/login', {
      email: email.trim(),
      password: password.trim(),
    });

    const token = res.data.access_token;
    const username = res.data.username; // assuming returned now

    if (typeof window !== 'undefined') {
      localStorage.setItem('token', token);
      localStorage.setItem('username', username);
    }

    // 👉 Now fetch latest conversation and go to it
    const convoRes = await axios.get('http://localhost:8000/auth/conversations', {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (convoRes.data.length > 0) {
      const latest = convoRes.data[0];
      navigate(`/chat/${latest.id}`);
    } else {
      // Create a new one if none exist
      const newRes = await axios.post('http://localhost:8000/auth/conversations', {}, {
        headers: { Authorization: `Bearer ${token}` },
      });
      navigate(`/chat/${newRes.data.id}`);
    }
  } catch (error) {
    console.error('Login error:', error.response?.data || error.message);
    if (error.response?.status === 422) {
      const details = error.response.data.detail;
      if (Array.isArray(details)) {
        setStatus(`Validation error: ${details.map(d => d.msg).join(', ')}`);
      } else {
        setStatus('Please check your email format and try again.');
      }
    } else {
      setStatus('Login failed. Please check your credentials.');
    }
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
          display: flex;
          max-width: 1000px;
          width: 90%;
          background-color: white;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
          border-radius: 10px;
          overflow: hidden;
        }
        .login-form {
          width: 50%;
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
        }
        .submit-btn:hover {
          background: linear-gradient(to right, rgb(91, 33, 182), rgb(79, 70, 229));
        }
        .options-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 16px;
          font-size: 14px;
          color: #6b7280;
        }
        .link-btn {
          background: none;
          border: none;
          color: rgb(109, 40, 217);
          cursor: pointer;
          text-decoration: underline;
        }
        .error-msg {
          margin-top: 16px;
          color: #dc2626;
          font-size: 14px;
        }
        .welcome-panel {
          width: 50%;
          background: linear-gradient(to bottom right, rgb(109, 40, 217), rgb(91, 33, 182));
          color: white;
          padding: 40px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }
        .signup-link {
          margin-top: 12px;
          padding: 10px 20px;
          border: 1px solid white;
          color: white;
          border-radius: 25px;
          text-decoration: none;
          transition: all 0.3s ease;
        }
        .signup-link:hover {
          background-color: white;
          color: rgb(109, 40, 217);
        }
      `}</style>

      <div className="login-box">
        <div className="login-form">
          <h1 className="brand">Mindora Bot</h1>
          <h2 className="form-title">Sign In</h2>

          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            className="input-field"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            className="input-field"
          />

          <button onClick={handleLogin} className="submit-btn" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>

          <div className="options-row">
            <label>
              <input type="checkbox" className="checkbox" /> Remember Me
            </label>
            <button onClick={() => alert('Feature coming soon')} className="link-btn">
              Forgot Password?
            </button>
          </div>

          {status && <p className="error-msg">{status}</p>}
        </div>

        <div className="welcome-panel">
          <h2>Welcome back</h2>
          <p>Don’t have an account?</p>
          <a href="/register" className="signup-link">Sign Up</a>
        </div>
      </div>
    </div>
  );
}
