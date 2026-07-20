import { useCallback } from 'react';
import axios from 'axios';

// Same pattern as src/api/api.js — read the backend URL from the build-time
// env var instead of hardcoding localhost, so this works in production too.
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Custom hook for API operations
const useChatAPI = () => {
  const token = localStorage.getItem('token') ?? '';

  const handleError = useCallback((err, fallbackMessage) => {
    console.error(fallbackMessage, err);
    if (err.response?.status === 401) {
      alert("Session expired. Please log in again.");
    }
  }, []);

  const fetchConversations = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/auth/conversations`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      return res.data;
    } catch (err) {
      handleError(err, 'Failed to fetch conversations');
      return [];
    }
  }, [token, handleError]);

  const fetchMessages = useCallback(async (conversationId) => {
    try {
      const res = await axios.get(`${API_BASE}/auth/conversations/${conversationId}/messages`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      return res.data;
    } catch (err) {
      handleError(err, 'Failed to fetch messages');
      return [];
    }
  }, [token, handleError]);

  const createConversation = useCallback(async () => {
    try {
      const res = await axios.post(
        `${API_BASE}/auth/conversations`,
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );
      return res.data;
    } catch (err) {
      handleError(err, 'Failed to create new chat');
      return null;
    }
  }, [token, handleError]);

  const deleteConversation = useCallback(async (chatId) => {
    try {
      await axios.delete(`${API_BASE}/auth/conversations/${chatId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      return true;
    } catch (err) {
      handleError(err, 'Failed to delete conversation');
      return false;
    }
  }, [token, handleError]);

  return {
    fetchConversations,
    fetchMessages,
    createConversation,
    deleteConversation,
    handleError
  };
};

export default useChatAPI;