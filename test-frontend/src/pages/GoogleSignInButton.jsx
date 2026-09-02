import { useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { googleAuth, getChats, startNewChat } from '../api/api';

const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID;

// Renders Google's own "Sign in with Google" button via Google Identity
// Services (loaded as a <script> in public/index.html) and handles both
// login and signup through the same backend endpoint — the backend decides
// whether this is a new or returning account.
export default function GoogleSignInButton({ onError }) {
  const buttonRef = useRef(null);
  const navigate = useNavigate();

  const handleCredentialResponse = useCallback(async (response) => {
    try {
      const res = await googleAuth(response.credential);
      localStorage.setItem('token', res.data.access_token);
      localStorage.setItem('user_id', res.data.user_id);
      localStorage.setItem('username', res.data.username);
      localStorage.setItem('gender', res.data.gender || '');

      // Could be a brand-new account or a returning one — go to their
      // latest conversation if they have one, otherwise start a fresh one.
      const convRes = await getChats();
      if (convRes.data.length > 0) {
        navigate(`/chat/${convRes.data[0].id}`);
      } else {
        const newRes = await startNewChat();
        navigate(`/chat/${newRes.data.id}`);
      }
    } catch (err) {
      console.error('Google sign-in error:', err.response?.data || err.message);
      onError?.('Google sign-in failed. Please try again.');
    }
  }, [navigate, onError]);

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) {
      console.warn('REACT_APP_GOOGLE_CLIENT_ID is not set — Google sign-in button will not render.');
      return;
    }

    let cancelled = false;
    let pollInterval = null;

    const renderButton = () => {
      if (cancelled || !buttonRef.current || !window.google?.accounts?.id) return;
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: handleCredentialResponse,
      });
      window.google.accounts.id.renderButton(buttonRef.current, {
        theme: 'outline',
        size: 'large',
        width: 320,
        text: 'continue_with',
      });
    };

    if (window.google?.accounts?.id) {
      renderButton();
    } else {
      // The GIS script tag loads with `defer` — poll briefly until it's ready
      // rather than assuming it's already there on first render.
      pollInterval = setInterval(() => {
        if (window.google?.accounts?.id) {
          clearInterval(pollInterval);
          renderButton();
        }
      }, 100);
    }

    return () => {
      cancelled = true;
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [handleCredentialResponse]);

  if (!GOOGLE_CLIENT_ID) return null;

  return <div ref={buttonRef} className="google-signin-btn" />;
}
