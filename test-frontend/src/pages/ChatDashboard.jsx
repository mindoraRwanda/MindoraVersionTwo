import React, { useEffect, useState, useRef, useCallback } from 'react';
import { sendVoiceMessage, streamMessageResponse } from '../api/api';
import Message from './Message';
import Sidebar from './Sidebar';
import WelcomeScreen from './WelcomeScreen';
import DeleteConfirmModal from './DeleteConfirmModal';
import useChatAPI from './useChatAPI';
import { logout } from '../utils/auth';
import './ChatDashboard.css';

const groupMessagesByDate = (messages) => {
  const grouped = (Array.isArray(messages) ? messages : []).reduce((acc, msg) => {
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
  const [deleteModal, setDeleteModal] = useState(null); // { id, label } | null
  const [isDeleting, setIsDeleting] = useState(false);
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('darkMode') === '1');

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
  const textareaRef = useRef(null);

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
      setMessages(Array.isArray(messagesData) ? messagesData : []);
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
    setHasStartedChat(false); // show welcome screen for the fresh chat
  }, [createConversation, fetchConversations]);

  const handleDeleteChat = useCallback((chatId, e) => {
    e.stopPropagation();
    const chat = conversations.find(c => c.id === chatId);
    const firstMsg = chat?.messages?.find(m => m.sender === 'user')?.content;
    const label = firstMsg
      ? firstMsg.slice(0, 30) + (firstMsg.length > 30 ? '…' : '')
      : `Started ${new Date(chat?.started_at).toLocaleDateString()}`;
    setDeleteModal({ id: chatId, label });
  }, [conversations]);

  const handleConfirmDelete = useCallback(async () => {
    if (!deleteModal) return;
    setIsDeleting(true);
    const success = await deleteConversation(deleteModal.id);
    setIsDeleting(false);
    setDeleteModal(null);
    if (success) {
      const wasSelected = selectedChat?.id === deleteModal.id;
      const conversationsData = await fetchConversations();
      setConversations(conversationsData);

      // If the deleted chat was open, or no chats remain → welcome screen
      if (wasSelected || conversationsData.length === 0) {
        // Pick the next available chat, or go to welcome state
        const next = conversationsData.find(c => c.id !== deleteModal.id) ?? null;
        setSelectedChat(next);
        setMessages([]);
        setHasStartedChat(false);
      }
    }
  }, [deleteModal, deleteConversation, fetchConversations, selectedChat]);

  const handleCancelDelete = useCallback(() => {
    if (!isDeleting) setDeleteModal(null);
  }, [isDeleting]);

  const handleLogout = useCallback(() => {
    logout();
  }, []);

  const toggleSidebar = useCallback(() => setSidebarVisible(prev => !prev), []);

  // -------- TEXT SEND (SSE streaming) --------
  const STREAMING_ID = '__streaming__';

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
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setLoading(true); // shows animated typing indicator

    try {
      const response = await streamMessageResponse(selectedChat.id, originalInput);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      // Dots stay visible until the first token arrives — then bubble is created and dots vanish
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalMeta = null;
      let bubbleCreated = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token !== undefined) {
              if (!bubbleCreated) {
                // First token: create the bubble with content already in it, then hide dots
                setMessages(prev => [...prev, {
                  id: STREAMING_ID,
                  sender: 'bot',
                  content: data.token,
                  timestamp: new Date().toISOString()
                }]);
                setLoading(false); // dots disappear exactly when text starts appearing
                bubbleCreated = true;
              } else {
                setMessages(prev => prev.map(m =>
                  m.id === STREAMING_ID
                    ? { ...m, content: m.content + data.token }
                    : m
                ));
              }
            } else if (data.done) {
              finalMeta = { id: data.id, timestamp: data.timestamp };
            }
          } catch { /* ignore malformed SSE lines */ }
        }
      }

      // Replace temporary id with the real DB id + timestamp
      if (bubbleCreated) {
        setMessages(prev => prev.map(m =>
          m.id === STREAMING_ID
            ? { ...m, id: finalMeta?.id ?? m.id, timestamp: finalMeta?.timestamp ?? m.timestamp }
            : m
        ));
      }

    } catch (err) {
      setMessages(prev => prev.filter(m => m.id !== STREAMING_ID));
      setInput(originalInput);
      handleError(err, 'Failed to send message');
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
      className={`chat-dashboard${darkMode ? ' dark' : ''}`}
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
          <button
            className="dark-toggle-btn"
            title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            onClick={() => setDarkMode(prev => {
              const next = !prev;
              localStorage.setItem('darkMode', next ? '1' : '0');
              return next;
            })}
          >
            {darkMode ? '☀️' : '🌙'}
          </button>
        </div>

        <div className="chat-content">
          <div className="chat-panel">
            <div className="messages">
              {error && <div className="error-message">{error}</div>}
              {isLoadingConversations && <div className="loading-text">Loading conversations...</div>}
              {!isLoadingConversations && !error && !hasStartedChat && messages.length === 0 && (
                <WelcomeScreen
                  onSuggestionClick={text => {
                    setInput(text);
                    setTimeout(() => textareaRef.current?.focus(), 0);
                  }}
                />
              )}

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

              {loading && (
                <div className="message bot typing-message">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              )}
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
                        ref={textareaRef}
                        className="input-box"
                        value={input}
                        onChange={e => {
                          setInput(e.target.value);
                          const ta = e.target;
                          ta.style.height = 'auto';
                          ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
                        }}
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

      {deleteModal && (
        <DeleteConfirmModal
          chatLabel={deleteModal.label}
          onConfirm={handleConfirmDelete}
          onCancel={handleCancelDelete}
          isDeleting={isDeleting}
        />
      )}
    </div>
  );
}