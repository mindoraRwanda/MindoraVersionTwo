import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { register, startNewChat } from '../api/api';


export default function Register() {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [gender, setGender] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleRegister = async () => {
  try {
    const res = await register(username, email, password, gender);
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', res.data.access_token);
      localStorage.setItem('user_id', res.data.user_id);
      localStorage.setItem('gender', res.data.gender || '');
    }

    // Create first conversation
    const conv = await startNewChat();
    const convId = conv.data.id;

    navigate(`/chat/${convId}`);
  } catch (err) {
    const issues = err.response?.data?.detail;
    if (Array.isArray(issues)) {
      const msg = issues.map(e => `• ${e.msg}`).join('\n');
      setError(msg);
    } else {
      setError('Registration failed. Try a different email.');
    }
  }
};


  return (
    <div className="register-container">
      <style>{`
        .register-container {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: #f3f4f6;
          font-family: sans-serif;
        }

        .register-box {
          display: flex;
          max-width: 1000px;
          width: 90%;
          background-color: white;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
          border-radius: 10px;
          overflow: hidden;
        }

        .register-form {
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

        .login-link {
          margin-top: 12px;
          padding: 10px 20px;
          border: 1px solid white;
          color: white;
          border-radius: 25px;
          text-decoration: none;
          transition: all 0.3s ease;
        }

        .login-link:hover {
          background-color: white;
          color: rgb(109, 40, 217);
        }
      `}</style>

      <div className="register-box">
        <div className="register-form">
          <h1 className="brand">Mindora Bot</h1>
          <h2 className="form-title">Create Account</h2>


          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            className="input-field"
          />

          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            className="input-field"
          />
          
          
          <select
            value={gender}
            onChange={e => setGender(e.target.value)}
            className="input-field"
          >
            <option value="">Select Gender (Optional)</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
            <option value="prefer_not_to_say">Prefer not to say</option>
          </select>

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            className="input-field"
          />


          <button onClick={handleRegister} className="submit-btn">
            Register
          </button>

          {error && <p className="error-msg">{error}</p>}
        </div>

        <div className="welcome-panel">
          <h2>Welcome to Mindora</h2>
          <p>Already have an account?</p>
          <a href="/" className="login-link">Sign In</a>
        </div>
      </div>
    </div>
  );
}
