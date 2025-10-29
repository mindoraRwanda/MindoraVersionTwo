// import axios from 'axios';

// const API_BASE = 'http://localhost:8000';

// const getAuthHeaders = () => {
//   const token = localStorage.getItem('token');
//   return { Authorization: `Bearer ${token}` };
// };

// // Auth
// export const login = (email, password) =>
//   axios.post(`${API_BASE}/auth/login`, { email, password });

// export const register = (username, email, password) =>
//   axios.post(`${API_BASE}/auth/signup`, {
//     username,
//     email,
//     password,
//   });

// // Conversations
// export const getChats = () =>
//   axios.get(`${API_BASE}/auth/conversations`, { headers: getAuthHeaders() });

// export const startNewChat = () =>
//   axios.post(`${API_BASE}/auth/conversations`, {}, { headers: getAuthHeaders() });

// // Messages
// export const sendMessage = async (conversation_id, content) => {
//   const res = await axios.post(
//     'http://localhost:8000/auth/messages',
//     { conversation_id, content },
//     {
//       headers: {
//         Authorization: `Bearer ${localStorage.getItem('token')}`
//       }
//     }
//   );

//   // Transform backend response to expected frontend format
//   return {
//     response: {
//       content: res.data.content,
//       timestamp: res.data.timestamp
//     },
//     emotion: null // Backend doesn't return emotion in message response
//   };
// };



// export const getMessages = (conversation_id) =>
//   axios.get(`${API_BASE}/auth/conversations/${conversation_id}/messages`, { headers: getAuthHeaders() });



// // Context Window
// export const fetchChatContext = async (limit = 10) => {
//   try {
//     const response = await axios.get(`${API_BASE}/auth/context`, {
//       params: { limit },
//       headers: getAuthHeaders(),
//     });
//     return response.data;
//   } catch (error) {
//     console.error('Error fetching chat context:', error);
//     return [];
//   }
// };

import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return { Authorization: `Bearer ${token}` };
};

// ---------- Auth ----------
export const login = (email, password) =>
  axios.post(`${API_BASE}/auth/login`, { email, password });

export const register = (username, email, password) =>
  axios.post(`${API_BASE}/auth/signup`, { username, email, password });

// ---------- Conversations ----------
export const getChats = () =>
  axios.get(`${API_BASE}/auth/conversations`, { headers: getAuthHeaders() });

export const startNewChat = () =>
  axios.post(`${API_BASE}/auth/conversations`, {}, { headers: getAuthHeaders() });

// ---------- Text Messages ----------
export const sendMessage = async (conversation_id, content) => {
  const res = await axios.post(
    `${API_BASE}/auth/messages`,                     // <- use API_BASE here
    { conversation_id, content },
    { headers: getAuthHeaders() }
  );

  // Normalize to the shape your ChatDashboard expects
  return {
    response: {
      content: res.data.content,
      timestamp: res.data.timestamp
    },
    emotion: null
  };
};

// ---------- Voice Messages (new) ----------
/**
 * Upload a recorded audio blob for STT → normal message pipeline.
 * Expects backend route: POST /auth/messages/voice (multipart/form-data)
 * Form fields:
 *   - audio: file (Blob)
 *   - conversation_id: number|string
 *   - meta: JSON string (optional) e.g. { mimeType, durationMs }
 *
 * Recommended backend response:
 * {
 *   transcript: "transcribed text",
 *   response: { content: "bot reply", timestamp: "..." },
 *   emotion: "anxiety" // optional
 * }
 */
export const sendVoiceMessage = async (conversation_id, audioBlob, meta = {}) => {
  const form = new FormData();
  // Use a filename that matches your server's expectation (e.g., 'audio')
  form.append('file', audioBlob, 'recording.webm');
  form.append('conversation_id', String(conversation_id));
  form.append('meta', JSON.stringify(meta));

  const res = await axios.post(
    `${API_BASE}/voice/messages`,
    form,
    {
      headers: {
        ...getAuthHeaders(),
        'Content-Type': 'multipart/form-data'
      },
      // Optional: increase if uploads are large
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    }
  );

  // Normalize possible shapes to what ChatDashboard uses
  const data = res.data || {};
  const responseContent =
    data?.response?.content ??
    data?.content ??                 // fallback if server returns like /auth/messages
    '';
  const responseTimestamp =
    data?.response?.timestamp ??
    data?.timestamp ??
    new Date().toISOString();

  return {
    transcript: data.transcript || null,
    response: {
      content: responseContent,
      timestamp: responseTimestamp,
    },
    emotion: data.emotion ?? null,
  };
};

// ---------- History ----------
export const getMessages = (conversation_id) =>
  axios.get(`${API_BASE}/auth/conversations/${conversation_id}/messages`, {
    headers: getAuthHeaders(),
  });

// ---------- Context Window ----------
export const fetchChatContext = async (limit = 10) => {
  try {
    const response = await axios.get(`${API_BASE}/auth/context`, {
      params: { limit },
      headers: getAuthHeaders(),
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching chat context:', error);
    return [];
  }
};

