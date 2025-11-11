import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return { Authorization: `Bearer ${token}` };
};

// Auth
export const login = (email, password) =>
  axios.post(`${API_BASE}/auth/login`, { email, password });

export const register = (username, email, password, gender) =>
  axios.post(`${API_BASE}/auth/signup`, {
    username,
    email,
    password,
    gender: gender || undefined,
  });

// Conversations
export const getChats = () =>
  axios.get(`${API_BASE}/auth/conversations`, { headers: getAuthHeaders() });

export const startNewChat = () =>
  axios.post(`${API_BASE}/auth/conversations`, {}, { headers: getAuthHeaders() });

// Messages
export const sendMessage = async (conversation_id, content) => {
  const res = await axios.post(
    'http://localhost:8000/auth/messages',
    { conversation_id, content },
    {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    }
  );

  // Debug: Log raw backend response
  console.log('🔍 RAW BACKEND RESPONSE:', res.data);

  // Transform backend response to expected frontend format
  return {
    response: {
      content: res.data.content,
      timestamp: res.data.timestamp,
      should_chunk: res.data.should_chunk || false,
      response_chunks: res.data.response_chunks || []
    },
    emotion: null // Backend doesn't return emotion in message response
  };
};



export const getMessages = (conversation_id) =>
  axios.get(`${API_BASE}/auth/conversations/${conversation_id}/messages`, { headers: getAuthHeaders() });



// Context Window
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
