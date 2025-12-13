import { useCallback } from 'react';
import axios from 'axios';

// Custom hook for API operations
const useChatAPI = () => {
  const handleError = useCallback((err, fallbackMessage) => {
    console.error(fallbackMessage, err);
    if (err.response?.status === 401) {
      alert("Session expired. Please log in again.");
    }
  }, []);

  const getToken = () => {
    return typeof window !== 'undefined' ? localStorage.getItem('token') ?? '' : '';
  };

  const fetchConversations = useCallback(async () => {
    try {
      const token = getToken();
      const res = await axios.get('http://localhost:8000/auth/conversations', {
        headers: { Authorization: `Bearer ${token}` },
      });
      return res.data;
    } catch (err) {
      handleError(err, 'Failed to fetch conversations');
      return [];
    }
  }, [handleError]);

  const fetchMessages = useCallback(async (conversationId) => {
    try {
      const token = getToken();
      const res = await axios.get(`http://localhost:8000/auth/conversations/${conversationId}/messages`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      return res.data;
    } catch (err) {
      handleError(err, 'Failed to fetch messages');
      return [];
    }
  }, [handleError]);

  const createConversation = useCallback(async () => {
    try {
      const token = getToken();
      const res = await axios.post(
        'http://localhost:8000/auth/conversations',
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );
      return res.data;
    } catch (err) {
      handleError(err, 'Failed to create new chat');
      return null;
    }
  }, [handleError]);

  const deleteConversation = useCallback(async (chatId) => {
    try {
      const token = getToken();
      await axios.delete(`http://localhost:8000/auth/conversations/${chatId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      return true;
    } catch (err) {
      handleError(err, 'Failed to delete conversation');
      return false;
    }
  }, [handleError]);

  return {
    fetchConversations,
    fetchMessages,
    createConversation,
    deleteConversation,
    handleError
  };
};

export default useChatAPI;