/**
 * Where the access token comes from.
 *
 * Two modes, decided automatically at load time:
 *
 *   EMBEDDED   The app is running inside an iframe on the main web app. It asks
 *              the parent page for a token and keeps it in memory only. Nothing
 *              is written to localStorage — an XSS here cannot walk off with a
 *              long-lived credential.
 *
 *   STANDALONE The app is running on its own, as it always has. The token comes
 *              from localStorage, put there by the existing login page.
 *
 * Standalone mode is untouched, so the current deployment keeps working while
 * the embedded integration is being tested.
 */

// The origin of the page allowed to embed this app and send it tokens.
// Set REACT_APP_PARENT_ORIGIN in .env to the main web app's origin.
const PARENT_ORIGIN = process.env.REACT_APP_PARENT_ORIGIN || 'http://localhost:4000';

const EMBEDDED = typeof window !== 'undefined' && window.parent !== window;

let memoryToken = null;
const readyWaiters = [];

/** True when running inside the main web app's iframe. */
export function isEmbedded() {
  return EMBEDDED;
}

/** The token to put in the Authorization header. */
export function getToken() {
  if (EMBEDDED) return memoryToken;
  return localStorage.getItem('token');
}

/** Authorization header, or an empty object when there is no token yet. */
export function authHeader() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/**
 * Resolves once a token is available. In standalone mode that is immediate;
 * embedded, it waits for the parent to answer the handshake.
 */
export function whenTokenReady() {
  if (getToken()) return Promise.resolve(getToken());
  if (!EMBEDDED) return Promise.resolve(null);
  return new Promise((resolve) => readyWaiters.push(resolve));
}

/**
 * Ask the parent for a new token. Call this after a 401 — the parent fetches a
 * fresh one from its own backend and posts it back.
 */
export function requestFreshToken() {
  if (!EMBEDDED) return;
  window.parent.postMessage({ type: 'MINDORA_TOKEN_EXPIRED' }, PARENT_ORIGIN);
}

/**
 * The user pressed the button that would be "Logout" in the standalone app.
 *
 * Embedded, this chat does not own the session and must not end it — the main
 * web app does. So we report the intent and let that app decide what happens:
 * navigate home, close the panel, sign the user out of everything. The chat
 * simply stops here.
 */
export function requestExit() {
  if (!EMBEDDED) return;
  window.parent.postMessage({ type: 'MINDORA_EXIT' }, PARENT_ORIGIN);
}

/**
 * Tell the parent the conversation was classified as a crisis, so it can show
 * the help resources full-screen instead of leaving them inside a small frame.
 */
export function notifyCrisis(detail = {}) {
  if (!EMBEDDED) return;
  window.parent.postMessage({ type: 'MINDORA_CRISIS', ...detail }, PARENT_ORIGIN);
}

if (EMBEDDED) {
  window.addEventListener('message', (event) => {
    // Only the configured parent may hand us a token.
    if (event.origin !== PARENT_ORIGIN) return;
    if (event.data?.type !== 'MINDORA_TOKEN') return;
    if (!event.data.token) return;

    memoryToken = event.data.token;
    while (readyWaiters.length) readyWaiters.shift()(memoryToken);
  });

  // Speak first. If we waited for the parent to send a token unprompted, a token
  // sent before this listener existed would be lost and the chat would sit blank.
  window.parent.postMessage({ type: 'MINDORA_READY' }, PARENT_ORIGIN);
}
