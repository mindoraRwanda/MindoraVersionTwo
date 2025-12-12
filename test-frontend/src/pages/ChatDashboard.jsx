import React, { useEffect, useState, useRef, useCallback } from 'react';
import { sendMessage, sendVoiceMessage } from '../api/api';
import Message from './Message';
import Sidebar from './Sidebar';
import WelcomeScreen from './WelcomeScreen';
import useChatAPI from './useChatAPI';
import './ChatDashboard.css';

const groupMessagesByDate = (messages) => {
  const grouped = messages.reduce((acc, msg) => {
    const date = new Date(msg.timestamp).toLocaleDateString([], { year: 'numeric', month: 'long', day: 'numeric' });
    if (!acc[date]) {
      acc[date] = [];
    }
    acc[date].push(msg);
    return acc;
  }, {});
  return Object.entries(grouped);
};

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

  // --- Voice state ---
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioURL, setAudioURL] = useState('');
  const [recordError, setRecordError] = useState(null);
  const [recordDurationMs, setRecordDurationMs] = useState(0);

  const mediaRecorderRef = useRef(null);
  const mediaChunksRef = useRef([]);
  const recordStartedAtRef = useRef(0);

  const messagesEndRef = useRef(null);

  // Use custom hook for API operations
  const {
    fetchConversations,
    fetchMessages,
    createConversation,
    deleteConversation,
    handleError
  } = useChatAPI();

  // Load conversations
  useEffect(() => {
    const loadConversations = async () => {
      try {
        setIsLoadingConversations(true);
        setError(null);
        const conversationsData = await fetchConversations();
        setConversations(conversationsData);
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
    if (!selectedChat) return;
    (async () => {
      const messagesData = await fetchMessages(selectedChat.id);
      setMessages(messagesData);
    })();
  }, [fetchMessages, selectedChat]);

  const handleSelectChat = useCallback((chat) => {
    setSelectedChat(chat);
    if (sidebarVisible) setSidebarVisible(false);
  }, [sidebarVisible]);

  const handleNewChat = useCallback(async () => {
    const newChatData = await createConversation();
    if (!newChatData) return;
    const conversationsData = await fetchConversations();
    setConversations(conversationsData);
    const newChat = { id: newChatData.id, started_at: newChatData.started_at };
    setSelectedChat(newChat);
    setMessages([]);
  }, [createConversation, fetchConversations]);

  const handleDeleteChat = useCallback(async (chatId, e) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this conversation?')) return;

    const success = await deleteConversation(chatId);
    if (success) {
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

  const toggleSidebar = useCallback(() => setSidebarVisible(prev => !prev), []);

  // -------- TEXT SEND --------
  const handleSend = useCallback(async () => {
    if (!input.trim() || !selectedChat) return;

    if (!hasStartedChat) setHasStartedChat(true);

    const userMsg = {
      sender: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMsg]);
    const originalInput = input;
    setInput('');
    setLoading(true);

    try {
      const res = await sendMessage(selectedChat.id, originalInput);
      console.log('res', res);
      if (!res?.response?.content) throw new Error('Invalid bot response');

      const botMsg = {
        sender: 'bot',
        content: res.response.content,
        timestamp: res.response.timestamp || new Date().toISOString(),
        emotion: res.emotion || null
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      handleError(err, 'Failed to send message');
      // revert UI
      setMessages(prev => prev.filter(m => m !== userMsg));
      setInput(originalInput);
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

  // -------- VOICE RECORDING --------
  const startRecording = useCallback(async () => {
    if (!selectedChat) return;
    setRecordError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mr;
      mediaChunksRef.current = [];
      recordStartedAtRef.current = Date.now();

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) mediaChunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(mediaChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        setAudioURL(URL.createObjectURL(blob));
        setRecordDurationMs(Date.now() - recordStartedAtRef.current);
        // stop tracks
        stream.getTracks().forEach(t => t.stop());
      };

      mr.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Mic error:', err);
      setRecordError('Microphone permission is required to record.');
    }
  }, [selectedChat]);

  const stopRecording = useCallback(() => {
    try {
      mediaRecorderRef.current?.stop();
    } catch {}
    setIsRecording(false);
  }, []);

  const discardRecording = useCallback(() => {
    setAudioBlob(null);
    if (audioURL) URL.revokeObjectURL(audioURL);
    setAudioURL('');
    setRecordDurationMs(0);
    setRecordError(null);
  }, [audioURL]);

  const sendVoice = useCallback(async () => {
    if (!audioBlob || !selectedChat) return;

    if (!hasStartedChat) setHasStartedChat(true);
    setLoading(true);

    // (Optional) show a placeholder “voice message sent” bubble
    const voicePlaceholder = {
      sender: 'user',
      content: '🎤 (voice message)',
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, voicePlaceholder]);

    try {
      const meta = { mimeType: audioBlob.type || 'audio/webm', durationMs: recordDurationMs };
      const res = await sendVoiceMessage(selectedChat.id, audioBlob, meta);

      // Replace placeholder with transcript (if present)
      const transcriptText = res.transcript || '(no transcript)';
      setMessages(prev => {
        const withoutPlaceholder = prev.filter(m => m !== voicePlaceholder);
        return [
          ...withoutPlaceholder,
          {
            sender: 'user',
            content: `🎤 ${transcriptText}`,
            timestamp: new Date().toISOString()
          }
        ];
      });

      // Then add bot reply
      let botAudioURL = null;
      if (res.audio_content) {
        try {
          const byteCharacters = atob(res.audio_content);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], {type: 'audio/mpeg'});
          botAudioURL = URL.createObjectURL(blob);
        } catch (e) {
          console.error("Failed to decode or play audio content:", e);
        }
      }

      const botMsg = {
        sender: 'bot',
        content: res?.response?.content || 'Sorry, I could not process the audio.',
        timestamp: res?.response?.timestamp || new Date().toISOString(),
        emotion: res?.emotion || null,
        audioUrl: botAudioURL, // Add audioUrl to bot message
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      handleError(err, 'Failed to send voice message');
      // remove placeholder
      setMessages(prev => prev.filter(m => m !== voicePlaceholder));
    } finally {
      setLoading(false);
      // clear the clip
      discardRecording();
    }
  }, [audioBlob, selectedChat, recordDurationMs, handleError, hasStartedChat, discardRecording]);

  // Auto-scroll
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
              {error && <div className="error-message">{error}</div>}
              {isLoadingConversations && <div className="loading-text">Loading conversations...</div>}
              {!isLoadingConversations && !error && !hasStartedChat && messages.length === 0 && <WelcomeScreen />}

              {groupMessagesByDate(messages).map(([date, messagesForDate]) => (
                <React.Fragment key={date}>
                  <div className="date-header">{date}</div>
                  {messagesForDate.map((msg, i) => (
                    <Message
                      key={i}
                      message={msg}
                      isUser={msg.sender === 'user'}
                    />
                  ))}
                </React.Fragment>
              ))}

              {loading && <div className="message bot">Mindora is typing...</div>}
              <div ref={messagesEndRef} />
            </div>

            {selectedChat && (
              <div className="input-bar">
                {/* Mic / Recording controls */}
                <button
                  className={`mic-btn ${isRecording ? 'recording' : ''}`}
                  title={isRecording ? 'Stop recording' : 'Record voice'}
                  onClick={isRecording ? stopRecording : startRecording}
                >
                  {isRecording ? '⏹️' : '🎙️'}
                </button>

                <div className="input-stack">
                  {/* If we have a recorded clip, show audio preview + send/discard */}
                  {audioURL ? (
                    <div className="audio-preview">
                      <audio src={audioURL} controls preload="metadata" />
                      <div className="audio-actions">
                        <button className="send-voice-btn" onClick={sendVoice} disabled={loading}>
                          Send voice
                        </button>
                        <button className="discard-voice-btn" onClick={discardRecording} disabled={loading}>
                          Discard
                        </button>
                      </div>
                      {recordError && <div className="error-message">{recordError}</div>}
                    </div>
                  ) : (
                    <>
                      {/* Normal text input when no voice clip waiting */}
                      <textarea
                        className="input-box"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
                        rows={1}
                      />
                    </>
                  )}
                </div>

                {/* Send text button (hidden/disabled if a voice clip is present) */}
                <button
                  className="send-btn"
                  onClick={handleSend}
                  disabled={loading || !!audioURL || !input.trim()}
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