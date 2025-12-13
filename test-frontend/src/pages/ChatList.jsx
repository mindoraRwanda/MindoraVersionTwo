import { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

export default function ChatList() {
  const [conversations, setConversations] = useState([]);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    
    const fetchChats = async () => {
      try {
        const res = await axios.get('http://localhost:8000/auth/conversations', {
          headers: { Authorization: `Bearer ${token}` },
        });
        setConversations(res.data);
      } catch (err) {
        console.error(err);
        if (err.response?.status === 401) {
          setError('Session expired or unauthorized. Please log in again.');
        } else {
          setError('Failed to load conversations.');
        }
      }
    };

    fetchChats();
  }, []);

  const handleNewChat = async () => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    
    try {
      const res = await axios.post(
        'http://localhost:8000/auth/conversations',
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );
      navigate(`/chat/${res.data.conversation_id}`);

      // Refresh conversation list
      const convoRes = await axios.get('http://localhost:8000/auth/conversations', {
        headers: { Authorization: `Bearer ${token}` },
      });
      setConversations(convoRes.data);
    } catch (err) {
      console.error(err);
      setError('Could not start a new conversation.');
    }
  };

  const handleDeleteChat = async (chatId, e) => {
    e.stopPropagation(); // Prevent navigation when clicking delete
    
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    
    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return;
    }

    try {
      await axios.delete(`http://localhost:8000/auth/conversations/${chatId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      // Refresh conversation list
      const convoRes = await axios.get('http://localhost:8000/auth/conversations', {
        headers: { Authorization: `Bearer ${token}` },
      });
      setConversations(convoRes.data);
    } catch (err) {
      console.error(err);
      setError('Could not delete the conversation.');
    }
  };

  return (
    <div className="chatlist-container">
      <style>{`
        .chatlist-container {
          min-height: 100vh;
          padding: 40px;
          font-family: sans-serif;
          background-color: #f3f4f6;
        }

        .chatlist-header {
          font-size: 28px;
          color: #1e3a8a;
          margin-bottom: 20px;
        }

        .new-chat-btn {
          padding: 10px 20px;
          font-weight: bold;
          background: linear-gradient(to right, #1d4ed8, #1e3a8a);
          color: white;
          border: none;
          border-radius: 20px;
          cursor: pointer;
          margin-bottom: 20px;
        }

        .new-chat-btn:hover {
          background: linear-gradient(to right, #1e40af, #1e3a8a);
        }

        .error-msg {
          color: #dc2626;
          margin-bottom: 10px;
        }

        .chat-list {
          list-style: none;
          padding: 0;
        }

        .chat-list li {
          margin-bottom: 10px;
        }

        .chat-item {
          display: flex;
          align-items: center;
          gap: 10px;
          background-color: white;
          border: 1px solid #cbd5e1;
          border-radius: 10px;
          transition: background-color 0.3s;
        }

        .chat-item:hover {
          background-color: #e0e7ff;
        }

        .chat-button {
          flex: 1;
          padding: 10px 16px;
          background: none;
          border: none;
          cursor: pointer;
          text-align: left;
        }

        .delete-button {
          padding: 8px 12px;
          margin-right: 8px;
          background-color: #dc2626;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 12px;
          transition: background-color 0.3s;
        }

        .delete-button:hover {
          background-color: #b91c1c;
        }

        .no-chats {
          color: #475569;
          font-size: 14px;
        }
      `}</style>

      <h2 className="chatlist-header">Your Conversations</h2>
      <button className="new-chat-btn" onClick={handleNewChat}>
        + Start New Chat
      </button>

      {error && <p className="error-msg">{error}</p>}

      {conversations.length === 0 && !error && (
        <p className="no-chats">You have no conversations yet. Start one to begin chatting.</p>
      )}

      <ul className="chat-list">
        {conversations.map(chat => (
          <li key={chat.id}>
            <div className="chat-item">
              <button
                className="chat-button"
                onClick={() => navigate(`/chat/${chat.id}`)}
              >
                Conversation #{chat.id}
              </button>
              <button
                className="delete-button"
                onClick={(e) => handleDeleteChat(chat.id, e)}
                title="Delete conversation"
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
