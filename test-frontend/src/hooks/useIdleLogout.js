import { useEffect, useRef } from 'react';
import { logout } from '../utils/auth';

const IDLE_TIMEOUT_MS = 3 * 60 * 1000; // 3 minutes
const ACTIVITY_EVENTS = ['mousedown', 'mousemove', 'keydown', 'touchstart', 'scroll', 'wheel'];

// Logs the user out after IDLE_TIMEOUT_MS of no activity, for privacy on
// shared/public devices. The token check happens at timeout time (not at
// mount time) so this stays correct across login/logout without remounting.
export default function useIdleLogout() {
  const timerRef = useRef(null);

  useEffect(() => {
    const resetTimer = () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        if (localStorage.getItem('token')) {
          logout('idle');
        }
      }, IDLE_TIMEOUT_MS);
    };

    ACTIVITY_EVENTS.forEach(event =>
      window.addEventListener(event, resetTimer, { passive: true })
    );
    resetTimer();

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      ACTIVITY_EVENTS.forEach(event =>
        window.removeEventListener(event, resetTimer)
      );
    };
  }, []);
}
