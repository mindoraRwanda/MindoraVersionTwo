import { isEmbedded, requestExit } from './chatToken';

export function logout(reason) {
  // Embedded in the main web app: that app owns the session. Clearing storage
  // here would do nothing useful, and navigating to '/' would load the chat's
  // own login page *inside the iframe* — which is exactly the page that should
  // no longer exist. Hand the decision to the parent instead.
  if (isEmbedded()) {
    requestExit();
    return;
  }

  localStorage.removeItem('token');
  localStorage.removeItem('username');
  if (reason) {
    sessionStorage.setItem('logoutReason', reason);
  }
  window.location.href = '/';
}
