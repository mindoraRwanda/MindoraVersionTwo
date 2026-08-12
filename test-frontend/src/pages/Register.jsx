import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { register, startNewChat } from '../api/api';

const GENDER_OPTIONS = [
  { value: '', label: 'Select Gender (Optional)' },
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
  { value: 'prefer_not_to_say', label: 'Prefer not to say' },
];

export default function Register() {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [gender, setGender] = useState('');
  const [error, setError] = useState(null);
  const [genderOpen, setGenderOpen] = useState(false);
  const genderRef = useRef(null);
  const navigate = useNavigate();

  // Close the custom gender dropdown when tapping/clicking outside it
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (genderRef.current && !genderRef.current.contains(e.target)) {
        setGenderOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    document.addEventListener('touchstart', handleOutsideClick);
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('touchstart', handleOutsideClick);
    };
  }, []);

  const selectedGenderLabel =
    GENDER_OPTIONS.find(o => o.value === gender)?.label || GENDER_OPTIONS[0].label;

  const handleRegister = async (e) => {
  if (e) e.preventDefault();
  try {
    const res = await register(username, email, password, gender);
    localStorage.setItem('token', res.data.access_token);
    localStorage.setItem('user_id', res.data.user_id);
    localStorage.setItem('gender', res.data.gender || '');

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
          box-sizing: border-box;
        }

        .register-form {
          width: 50%;
          padding: 40px;
          background-color: #fff;
          box-sizing: border-box;
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

        /* Custom gender dropdown — a real <select>'s open list is rendered
           natively by the OS/browser and can't be restyled or size-capped
           via CSS, which is what made it look oversized. This gives full
           control over both the closed trigger and the open options list. */
        .custom-select {
          position: relative;
          margin-bottom: 16px;
        }

        .custom-select-trigger {
          margin-bottom: 0;
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          text-align: left;
          cursor: pointer;
          color: #111827;
        }

        .placeholder-text {
          color: #6b7280;
        }

        .custom-select-arrow {
          color: #6b7280;
          font-size: 11px;
          margin-left: 8px;
          flex-shrink: 0;
        }

        .custom-select-options {
          position: absolute;
          top: calc(100% + 6px);
          left: 0;
          right: 0;
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 14px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
          list-style: none;
          margin: 0;
          padding: 6px;
          max-height: 220px;
          overflow-y: auto;
          box-sizing: border-box;
          z-index: 20;
        }

        .custom-select-option {
          width: 100%;
          text-align: left;
          padding: 10px 14px;
          border: none;
          background: none;
          border-radius: 10px;
          cursor: pointer;
          font-size: 16px;
          font-family: inherit;
          color: #111827;
          box-sizing: border-box;
        }

        .custom-select-option:hover,
        .custom-select-option.selected {
          background: #f3e8ff;
          color: rgb(109, 40, 217);
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
          box-sizing: border-box;
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

        /* ── Small screens: stack the two panels instead of squeezing them side by side ── */
        @media (max-width: 700px) {
          .register-box {
            flex-direction: column;
            width: 94%;
          }
          .register-form,
          .welcome-panel {
            width: 100%;
          }
          .register-form {
            padding: 28px 24px;
          }
          /* Swap the big gradient panel for a plain centered text line — a
             full-height colored box under the form wastes space and looks
             heavy on mobile. Plain block + text-align:center is simpler and
             more reliable here than flex row-centering for one short line. */
          .welcome-panel {
            display: block;
            background: none;
            padding: 4px 24px 24px;
            text-align: center;
          }
          .welcome-panel h2 {
            display: none;
          }
          .welcome-panel p {
            display: inline;
            margin: 0;
            color: #6b7280;
            font-size: 14px;
          }
          .login-link {
            display: inline;
            margin-left: 10px;
            border: none;
            color: #2563eb;
            text-decoration: none;
            font-size: 14px;
            padding: 0;
            background: none;
          }
        }

        @media (max-width: 380px) {
          .brand {
            font-size: 26px;
          }
          .form-title {
            font-size: 20px;
          }
          .register-form {
            padding: 24px 18px;
          }
        }
      `}</style>

      <div className="register-box">
        <div className="register-form">
          <h1 className="brand">Mindora Bot</h1>
          <h2 className="form-title">Create Account</h2>

          <form onSubmit={handleRegister}>
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


            <div className="custom-select" ref={genderRef}>
              <button
                type="button"
                className="input-field custom-select-trigger"
                onClick={() => setGenderOpen(prev => !prev)}
              >
                <span className={gender ? '' : 'placeholder-text'}>{selectedGenderLabel}</span>
                <span className="custom-select-arrow">{genderOpen ? '▲' : '▼'}</span>
              </button>

              {genderOpen && (
                <ul className="custom-select-options">
                  {GENDER_OPTIONS.map(opt => (
                    <li key={opt.value}>
                      <button
                        type="button"
                        className={`custom-select-option${gender === opt.value ? ' selected' : ''}`}
                        onClick={() => { setGender(opt.value); setGenderOpen(false); }}
                      >
                        {opt.label}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="input-field"
            />

            <button type="submit" className="submit-btn">
              Register
            </button>
          </form>

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
