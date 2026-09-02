/**
 * EXAMPLE ONLY — for the main web app's team.
 *
 * This file is not part of the chatbot. It shows the entire amount of work the
 * Node.js side has to do: call one endpoint with a shared key, hand the returned
 * token to the frontend. Copy it into the main app's repo and adapt.
 *
 * The chatbot team gives you two things:
 *   MINDORA_CHATBOT_URL      e.g. https://mindora-backend.up.railway.app
 *   MINDORA_INTEGRATION_KEY  a secret key — BACKEND ONLY, never send it to a browser
 *
 * You give the chatbot nothing. No shared database, no shared signing secret,
 * no user export.
 */

const CHATBOT_URL = process.env.MINDORA_CHATBOT_URL;
const INTEGRATION_KEY = process.env.MINDORA_INTEGRATION_KEY;

if (!CHATBOT_URL || !INTEGRATION_KEY) {
  throw new Error('MINDORA_CHATBOT_URL and MINDORA_INTEGRATION_KEY must be set');
}

/**
 * Exchange one of YOUR authenticated users for a chatbot access token.
 *
 * @param {{ id: string|number, email: string, username?: string, gender?: string }} user
 *   `id` must be stable for the life of the account — the chatbot keys all chat
 *   history off it, so an id that changes orphans that user's conversations.
 * @returns {Promise<{access_token: string, expires_in: number, user_id: string}>}
 */
async function getChatbotSession(user) {
  const res = await fetch(`${CHATBOT_URL}/integration/session`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Integration-Key': INTEGRATION_KEY,
    },
    body: JSON.stringify({
      external_id: String(user.id),
      email: user.email,
      username: user.username,
      gender: user.gender,
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Chatbot session request failed (${res.status}): ${detail}`);
  }
  return res.json();
}

/**
 * Expose it to your own frontend, behind your own login.
 *
 *   app.get('/api/chat/session', requireLogin, chatSessionRoute);
 *
 * `requireLogin` is your existing session middleware — it must populate req.user.
 * This route is the security boundary: everything past it trusts that the person
 * is who req.user says they are.
 */
async function chatSessionRoute(req, res) {
  if (!req.user) return res.status(401).json({ error: 'Not signed in' });
  try {
    const session = await getChatbotSession(req.user);
    // Return only what the browser needs. Never the integration key.
    res.json({
      accessToken: session.access_token,
      expiresIn: session.expires_in,
      chatbotUrl: CHATBOT_URL,
    });
  } catch (err) {
    console.error('[mindora]', err.message);
    res.status(502).json({ error: 'Chat is unavailable' });
  }
}

module.exports = { getChatbotSession, chatSessionRoute };

/* ---------------------------------------------------------------------------
 * Browser side, for reference. Fetch a token, refresh it before it expires,
 * and talk to the chatbot directly.
 *
 *   let token = null, expiresAt = 0;
 *
 *   async function chatToken() {
 *     if (token && Date.now() < expiresAt - 60_000) return token;
 *     const r = await fetch('/api/chat/session', { credentials: 'same-origin' });
 *     if (!r.ok) throw new Error('Could not start chat session');
 *     const s = await r.json();
 *     token = s.accessToken;
 *     expiresAt = Date.now() + s.expiresIn * 1000;
 *     return token;
 *   }
 *
 *   async function sendMessage(conversationId, content) {
 *     const res = await fetch(`${CHATBOT_URL}/auth/messages/stream`, {
 *       method: 'POST',
 *       headers: {
 *         'Content-Type': 'application/json',
 *         Authorization: `Bearer ${await chatToken()}`,
 *       },
 *       body: JSON.stringify({ conversation_id: conversationId, content }),
 *     });
 *     // res.body is an SSE stream: data: {"token":"..."} per chunk.
 *     return res;
 *   }
 *
 * A 401 mid-session means the token expired — clear it and retry once.
 * ------------------------------------------------------------------------- */
