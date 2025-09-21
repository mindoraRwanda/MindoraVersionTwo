import { useEffect, useState, useRef, useCallback } from 'react';
import { sendMessage } from '../api/api';
import Message from './Message';
import Sidebar from './Sidebar';
import WelcomeScreen from './WelcomeScreen';
import useChatAPI from './useChatAPI';
import './ChatDashboard.css';


export default function ChatDashboard() {
  const [conversations, setConversations] = useState([]);
  const [selectedChat, setSelectedChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [hasStartedChat, setHasStartedChat] = useState(false);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const messagesEndRef = useRef(null);

  // Use custom hook for API operations
  const {
    fetchConversations,
    fetchMessages,
    createConversation,
    deleteConversation,
    handleError
  } = useChatAPI();

  // Load conversations on component mount
  useEffect(() => {
    const loadConversations = async () => {
      try {
        setIsLoadingConversations(true);
        setError(null);
        const conversationsData = await fetchConversations();
        setConversations(conversationsData);

        // Don't automatically set hasStartedChat to true just because they have old conversations
        // They should still see the welcome screen and can choose to continue or start new

        if (conversationsData.length && !selectedChat) {
          setSelectedChat(conversationsData[0]);
        }
      } catch (err) {
        setError('Failed to load conversations. Please refresh the page.');
        console.error('Error loading conversations:', err);
      } finally {
        setIsLoadingConversations(false);
      }
    };
    loadConversations();
  }, [fetchConversations, selectedChat]);

  // Load messages when selected chat changes
  useEffect(() => {
    if (selectedChat) {
      const loadMessages = async () => {
        const messagesData = await fetchMessages(selectedChat.id);
        setMessages(messagesData);

        // Don't automatically show chat interface just because they have old messages
        // They should still see the welcome screen and choose to continue or start fresh
      };
      loadMessages();
    }
  }, [fetchMessages, selectedChat]);

  const handleSelectChat = useCallback((chat) => {
    setSelectedChat(chat);
    // Close sidebar after selecting a chat
    if (sidebarVisible) {
      setSidebarVisible(false);
    }
  }, [sidebarVisible]);

  const handleNewChat = useCallback(async () => {
    const newChatData = await createConversation();
    if (newChatData) {
      const conversationsData = await fetchConversations();
      setConversations(conversationsData);
      const newChat = { id: newChatData.id, started_at: newChatData.started_at };
      setSelectedChat(newChat);
      // Clear messages to show welcome screen for new chat
      setMessages([]);
      // Don't automatically show sidebar - let user toggle it manually
    }
  }, [createConversation, fetchConversations]);


  const handleDeleteChat = useCallback(async (chatId, e) => {
    e.stopPropagation();

    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return;
    }

    const success = await deleteConversation(chatId);
    if (success) {
      // If we deleted the currently selected chat, clear selection
      if (selectedChat?.id === chatId) {
        setSelectedChat(null);
        setMessages([]);
      }

      const conversationsData = await fetchConversations();
      setConversations(conversationsData);
    }
  }, [deleteConversation, fetchConversations, selectedChat]);

  const handleLogout = useCallback(() => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    window.location.href = '/';
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarVisible(prev => !prev);
  }, []);

  /**
    * Handles sending a message to the chatbot
    * @param {string} messageContent - The message content to send
    */
  const handleSend = useCallback(async () => {
    if (!input.trim() || !selectedChat) return;

    // Mark that the user has started chatting
    if (!hasStartedChat) {
      setHasStartedChat(true);
      // Don't automatically show sidebar - let user toggle it manually
    }

    const userMsg = {
      sender: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await sendMessage(selectedChat.id, input);

      if (!res || !res.response || !res.response.content) {
        throw new Error('Invalid bot response');
      }

      const botMsg = {
        sender: 'bot',
        content: res.response.content,
        timestamp: res.response.timestamp || new Date().toISOString(),
        emotion: res.emotion || null
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      handleError(err, 'Failed to send message');
      // Restore user message if sending failed
      setMessages(prev => prev.filter(msg => msg.timestamp !== userMsg.timestamp));
      setInput(input); // Restore the input
    } finally {
      setLoading(false);
    }
  }, [input, selectedChat, handleError, hasStartedChat]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);


  return (
    <div
      className="chat-dashboard"
      style={{
        '--sidebar-width': sidebarVisible ? '320px' : '0px',
        '--sidebar-shadow': sidebarVisible ? '2px 0 20px rgba(109, 40, 217, 0.15)' : 'none'
      }}
    >
      <div className="sidebar-container">
        <div className="sidebar">
          <div className="sidebar-content">
            <Sidebar
              conversations={conversations}
              selectedChat={selectedChat}
              onSelectChat={handleSelectChat}
              onNewChat={handleNewChat}
              onDeleteChat={handleDeleteChat}
              onLogout={handleLogout}
            />
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="header">
          <button className="hamburger-btn" onClick={toggleSidebar}>
            {sidebarVisible ? '✕' : '☰'}
          </button>
          <div className="header-title">MINDORA</div>
        </div>
        
        <div className="chat-content">
          <div className="chat-panel">
            <div className="messages">
              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}
              {isLoadingConversations && (
                <div className="loading-text">
                  Loading conversations...
                </div>
              )}
              {!isLoadingConversations && !error && !hasStartedChat && messages.length === 0 && (
                <WelcomeScreen />
              )}
              {messages.map((msg, i) => (
                <Message
                  key={i}
                  message={msg}
                  isUser={msg.sender === 'user'}
                />
              ))}
              {loading && (
                <div className="message bot">
                  Mindora is typing...
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {selectedChat && (
              <div className="input-bar">
                <textarea
                  className="input-box"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
                  rows={1}
                />
                <button
                  className="send-btn"
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                >
                  Send
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
