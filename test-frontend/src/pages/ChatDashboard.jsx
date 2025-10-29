
// import { useEffect, useState, useRef, useCallback } from 'react';
// import { sendMessage } from '../api/api';
// import Message from './Message';
// import Sidebar from './Sidebar';
// import WelcomeScreen from './WelcomeScreen';
// import useChatAPI from './useChatAPI';
// import './ChatDashboard.css';


// export default function ChatDashboard() {
//   const [conversations, setConversations] = useState([]);
//   const [selectedChat, setSelectedChat] = useState(null);
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [isLoadingConversations, setIsLoadingConversations] = useState(true);
//   const [hasStartedChat, setHasStartedChat] = useState(false);
//   const [sidebarVisible, setSidebarVisible] = useState(false);
//   const messagesEndRef = useRef(null);

//   // Use custom hook for API operations
//   const {
//     fetchConversations,
//     fetchMessages,
//     createConversation,
//     deleteConversation,
//     handleError
//   } = useChatAPI();

//   // Load conversations on component mount
//   useEffect(() => {
//     const loadConversations = async () => {
//       try {
//         setIsLoadingConversations(true);
//         setError(null);
//         const conversationsData = await fetchConversations();
//         setConversations(conversationsData);

//         // Don't automatically set hasStartedChat to true just because they have old conversations
//         // They should still see the welcome screen and can choose to continue or start new

//         if (conversationsData.length && !selectedChat) {
//           setSelectedChat(conversationsData[0]);
//         }
//       } catch (err) {
//         setError('Failed to load conversations. Please refresh the page.');
//         console.error('Error loading conversations:', err);
//       } finally {
//         setIsLoadingConversations(false);
//       }
//     };
//     loadConversations();
//   }, [fetchConversations, selectedChat]);

//   // Load messages when selected chat changes
//   useEffect(() => {
//     if (selectedChat) {
//       const loadMessages = async () => {
//         const messagesData = await fetchMessages(selectedChat.id);
//         setMessages(messagesData);

//         // Don't automatically show chat interface just because they have old messages
//         // They should still see the welcome screen and choose to continue or start fresh
//       };
//       loadMessages();
//     }
//   }, [fetchMessages, selectedChat]);

//   const handleSelectChat = useCallback((chat) => {
//     setSelectedChat(chat);
//     // Close sidebar after selecting a chat
//     if (sidebarVisible) {
//       setSidebarVisible(false);
//     }
//   }, [sidebarVisible]);

//   const handleNewChat = useCallback(async () => {
//     const newChatData = await createConversation();
//     if (newChatData) {
//       const conversationsData = await fetchConversations();
//       setConversations(conversationsData);
//       const newChat = { id: newChatData.id, started_at: newChatData.started_at };
//       setSelectedChat(newChat);
//       // Clear messages to show welcome screen for new chat
//       setMessages([]);
//       // Don't automatically show sidebar - let user toggle it manually
//     }
//   }, [createConversation, fetchConversations]);


//   const handleDeleteChat = useCallback(async (chatId, e) => {
//     e.stopPropagation();

//     if (!window.confirm('Are you sure you want to delete this conversation?')) {
//       return;
//     }

//     const success = await deleteConversation(chatId);
//     if (success) {
//       // If we deleted the currently selected chat, clear selection
//       if (selectedChat?.id === chatId) {
//         setSelectedChat(null);
//         setMessages([]);
//       }

//       const conversationsData = await fetchConversations();
//       setConversations(conversationsData);
//     }
//   }, [deleteConversation, fetchConversations, selectedChat]);

//   const handleLogout = useCallback(() => {
//     localStorage.removeItem('token');
//     localStorage.removeItem('username');
//     window.location.href = '/';
//   }, []);

//   const toggleSidebar = useCallback(() => {
//     setSidebarVisible(prev => !prev);
//   }, []);

//   /**
//     * Handles sending a message to the chatbot
//     * @param {string} messageContent - The message content to send
//     */
//   const handleSend = useCallback(async () => {
//     if (!input.trim() || !selectedChat) return;

//     // Mark that the user has started chatting
//     if (!hasStartedChat) {
//       setHasStartedChat(true);
//       // Don't automatically show sidebar - let user toggle it manually
//     }

//     const userMsg = {
//       sender: 'user',
//       content: input,
//       timestamp: new Date().toISOString()
//     };

//     setMessages(prev => [...prev, userMsg]);
//     setInput('');
//     setLoading(true);

//     try {
//       const res = await sendMessage(selectedChat.id, input);

//       if (!res || !res.response || !res.response.content) {
//         throw new Error('Invalid bot response');
//       }

//       const botMsg = {
//         sender: 'bot',
//         content: res.response.content,
//         timestamp: res.response.timestamp || new Date().toISOString(),
//         emotion: res.emotion || null
//       };

//       setMessages(prev => [...prev, botMsg]);
//     } catch (err) {
//       handleError(err, 'Failed to send message');
//       // Restore user message if sending failed
//       setMessages(prev => prev.filter(msg => msg.timestamp !== userMsg.timestamp));
//       setInput(input); // Restore the input
//     } finally {
//       setLoading(false);
//     }
//   }, [input, selectedChat, handleError, hasStartedChat]);

//   const handleKeyDown = useCallback((e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       handleSend();
//     }
//   }, [handleSend]);

//   // Auto-scroll to bottom when messages change
//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [messages]);


//   return (
//     <div
//       className="chat-dashboard"
//       style={{
//         '--sidebar-width': sidebarVisible ? '320px' : '0px',
//         '--sidebar-shadow': sidebarVisible ? '2px 0 20px rgba(109, 40, 217, 0.15)' : 'none'
//       }}
//     >
//       <div className="sidebar-container">
//         <div className="sidebar">
//           <div className="sidebar-content">
//             <Sidebar
//               conversations={conversations}
//               selectedChat={selectedChat}
//               onSelectChat={handleSelectChat}
//               onNewChat={handleNewChat}
//               onDeleteChat={handleDeleteChat}
//               onLogout={handleLogout}
//             />
//           </div>
//         </div>
//       </div>

//       <div className="main-content">
//         <div className="header">
//           <button className="hamburger-btn" onClick={toggleSidebar}>
//             {sidebarVisible ? '✕' : '☰'}
//           </button>
//           <div className="header-title">MINDORA</div>
//         </div>
        
//         <div className="chat-content">
//           <div className="chat-panel">
//             <div className="messages">
//               {error && (
//                 <div className="error-message">
//                   {error}
//                 </div>
//               )}
//               {isLoadingConversations && (
//                 <div className="loading-text">
//                   Loading conversations...
//                 </div>
//               )}
//               {!isLoadingConversations && !error && !hasStartedChat && messages.length === 0 && (
//                 <WelcomeScreen />
//               )}
//               {messages.map((msg, i) => (
//                 <Message
//                   key={i}
//                   message={msg}
//                   isUser={msg.sender === 'user'}
//                 />
//               ))}
//               {loading && (
//                 <div className="message bot">
//                   Mindora is typing...
//                 </div>
//               )}
//               <div ref={messagesEndRef} />
//             </div>

//             {selectedChat && (
//               <div className="input-bar">
//                 <textarea
//                   className="input-box"
//                   value={input}
//                   onChange={e => setInput(e.target.value)}
//                   onKeyDown={handleKeyDown}
//                   placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
//                   rows={1}
//                 />
//                 <button
//                   className="send-btn"
//                   onClick={handleSend}
//                   disabled={loading || !input.trim()}
//                 >
//                   Send
//                 </button>
//               </div>
//             )}
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

// Second Iteration
// import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
// import { sendMessage, sendVoiceMessage } from '../api/api';
// import Message from './Message';
// import Sidebar from './Sidebar';
// import WelcomeScreen from './WelcomeScreen';
// import useChatAPI from './useChatAPI';
// import './ChatDashboard.css';

// export default function ChatDashboard() {
//   const [conversations, setConversations] = useState([]);
//   const [selectedChat, setSelectedChat] = useState(null);
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [isLoadingConversations, setIsLoadingConversations] = useState(true);
//   const [hasStartedChat, setHasStartedChat] = useState(false);
//   const [sidebarVisible, setSidebarVisible] = useState(false);

//   // --- Voice state ---
//   const [isRecording, setIsRecording] = useState(false);
//   const [recordingUrl, setRecordingUrl] = useState(null);
//   const [recordingBlob, setRecordingBlob] = useState(null);
//   const [recordSecs, setRecordSecs] = useState(0);
//   const [sttText, setSttText] = useState('');       // transcript returned by backend
//   const [uploadingVoice, setUploadingVoice] = useState(false);

//   const mediaRecorderRef = useRef(null);
//   const chunksRef = useRef([]);
//   const timerRef = useRef(null);

//   const messagesEndRef = useRef(null);

//   // Use custom hook for API operations
//   const {
//     fetchConversations,
//     fetchMessages,
//     createConversation,
//     deleteConversation,
//     handleError
//   } = useChatAPI();

//   // Load conversations on component mount
//   useEffect(() => {
//     (async () => {
//       try {
//         setIsLoadingConversations(true);
//         setError(null);
//         const conversationsData = await fetchConversations();
//         setConversations(conversationsData);
//         if (conversationsData.length && !selectedChat) {
//           setSelectedChat(conversationsData[0]);
//         }
//       } catch (err) {
//         setError('Failed to load conversations. Please refresh the page.');
//         console.error('Error loading conversations:', err);
//       } finally {
//         setIsLoadingConversations(false);
//       }
//     })();
//   }, [fetchConversations, selectedChat]);

//   // Load messages when selected chat changes
//   useEffect(() => {
//     if (!selectedChat) return;
//     (async () => {
//       const messagesData = await fetchMessages(selectedChat.id);
//       setMessages(messagesData);
//     })();
//   }, [fetchMessages, selectedChat]);

//   const handleSelectChat = useCallback((chat) => {
//     setSelectedChat(chat);
//     if (sidebarVisible) setSidebarVisible(false);
//   }, [sidebarVisible]);

//   const handleNewChat = useCallback(async () => {
//     const newChatData = await createConversation();
//     if (newChatData) {
//       const conversationsData = await fetchConversations();
//       setConversations(conversationsData);
//       const newChat = { id: newChatData.id, started_at: newChatData.started_at };
//       setSelectedChat(newChat);
//       setMessages([]);
//     }
//   }, [createConversation, fetchConversations]);

//   const handleDeleteChat = useCallback(async (chatId, e) => {
//     e.stopPropagation();
//     if (!window.confirm('Delete this conversation?')) return;
//     const success = await deleteConversation(chatId);
//     if (success) {
//       if (selectedChat?.id === chatId) {
//         setSelectedChat(null);
//         setMessages([]);
//       }
//       const conversationsData = await fetchConversations();
//       setConversations(conversationsData);
//     }
//   }, [deleteConversation, fetchConversations, selectedChat]);

//   const handleLogout = useCallback(() => {
//     localStorage.removeItem('token');
//     localStorage.removeItem('username');
//     window.location.href = '/';
//   }, []);

//   const toggleSidebar = useCallback(() => setSidebarVisible(prev => !prev), []);

//   // ---------- TEXT SEND ----------
//   const handleSend = useCallback(async () => {
//     if (!input.trim() || !selectedChat) return;

//     if (!hasStartedChat) setHasStartedChat(true);

//     const userMsg = { sender: 'user', content: input, timestamp: new Date().toISOString() };
//     setMessages(prev => [...prev, userMsg]);
//     setInput('');
//     setLoading(true);

//     try {
//       const res = await sendMessage(selectedChat.id, userMsg.content);
//       if (!res || !res.response || !res.response.content) throw new Error('Invalid bot response');

//       const botMsg = {
//         sender: 'bot',
//         content: res.response.content,
//         timestamp: res.response.timestamp || new Date().toISOString(),
//         emotion: res.emotion || null
//       };
//       setMessages(prev => [...prev, botMsg]);
//     } catch (err) {
//       handleError(err, 'Failed to send message');
//       // rollback user message and restore input
//       setMessages(prev => prev.filter(m => m.timestamp !== userMsg.timestamp));
//       setInput(userMsg.content);
//     } finally {
//       setLoading(false);
//     }
//   }, [input, selectedChat, handleError, hasStartedChat]);

//   const handleKeyDown = useCallback((e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       handleSend();
//     }
//   }, [handleSend]);

//   // ---------- VOICE: Start/Stop ----------
//   const startRecording = useCallback(async () => {
//     if (!selectedChat) return;
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//       const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' });
//       chunksRef.current = [];
//       mr.ondataavailable = (e) => e.data && chunksRef.current.push(e.data);
//       mr.onstop = () => {
//         const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
//         const url = URL.createObjectURL(blob);
//         setRecordingBlob(blob);
//         setRecordingUrl(url);
//         // stop timer
//         if (timerRef.current) {
//           clearInterval(timerRef.current);
//           timerRef.current = null;
//         }
//       };
//       mr.start(100);
//       setSttText('');
//       setRecordSecs(0);
//       setIsRecording(true);
//       mediaRecorderRef.current = mr;

//       // simple duration timer
//       timerRef.current = setInterval(() => {
//         setRecordSecs((s) => s + 1);
//       }, 1000);
//     } catch (e) {
//       console.error('Mic error:', e);
//       alert('Microphone permission is required to record.');
//     }
//   }, [selectedChat]);

//   const stopRecording = useCallback(() => {
//     if (!isRecording || !mediaRecorderRef.current) return;
//     mediaRecorderRef.current.stop();
//     mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
//     setIsRecording(false);
//   }, [isRecording]);

//   const discardRecording = useCallback(() => {
//     if (recordingUrl) URL.revokeObjectURL(recordingUrl);
//     setRecordingUrl(null);
//     setRecordingBlob(null);
//     setRecordSecs(0);
//     setSttText('');
//   }, [recordingUrl]);

//   const insertTranscriptIntoInput = useCallback(() => {
//     if (sttText) setInput(prev => (prev ? `${prev}\n${sttText}` : sttText));
//   }, [sttText]);

//   const handleSendVoice = useCallback(async () => {
//     if (!recordingBlob || !selectedChat) return;
//     setUploadingVoice(true);

//     // show a temporary “user voice message” bubble
//     const tempId = `voice-${Date.now()}`;
//     setMessages(prev => [
//       ...prev,
//       { sender: 'user', content: '[Voice message]', timestamp: tempId, audioUrl: recordingUrl }
//     ]);

//     try {
//       const res = await sendVoiceMessage(selectedChat.id, recordingBlob, {
//         mimeType: recordingBlob.type || 'audio/webm',
//         duration: recordSecs
//       });

//       // Backend returns: { content, timestamp, transcript? }
//       const transcript = res?.transcript || '';
//       if (transcript) setSttText(transcript);

//       const botMsg = {
//         sender: 'bot',
//         content: res?.response?.content || '(no response)',
//         timestamp: res?.response?.timestamp || new Date().toISOString()
//       };
//       // replace temp bubble (by timestamp) with final + bot
//       setMessages(prev => {
//         const withoutTemp = prev.filter(m => m.timestamp !== tempId);
//         return [...withoutTemp, { sender: 'user', content: transcript || '[Voice message]', timestamp: new Date().toISOString() }, botMsg];
//       });

//       // clear the clip after sending
//       discardRecording();
//       if (!hasStartedChat) setHasStartedChat(true);
//     } catch (err) {
//       handleError(err, 'Failed to send voice message');
//       // remove temp bubble
//       setMessages(prev => prev.filter(m => m.timestamp !== tempId));
//     } finally {
//       setUploadingVoice(false);
//     }
//   }, [recordingBlob, recordingUrl, selectedChat, recordSecs, discardRecording, handleError, hasStartedChat]);

//   // Auto-scroll
//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [messages]);

//   const formattedTime = useMemo(() => {
//     const m = Math.floor(recordSecs / 60).toString().padStart(2, '0');
//     const s = (recordSecs % 60).toString().padStart(2, '0');
//     return `${m}:${s}`;
//   }, [recordSecs]);

//   return (
//     <div
//       className="chat-dashboard"
//       style={{
//         '--sidebar-width': sidebarVisible ? '320px' : '0px',
//         '--sidebar-shadow': sidebarVisible ? '2px 0 20px rgba(109, 40, 217, 0.15)' : 'none'
//       }}
//     >
//       <div className="sidebar-container">
//         <div className="sidebar">
//           <div className="sidebar-content">
//             <Sidebar
//               conversations={conversations}
//               selectedChat={selectedChat}
//               onSelectChat={handleSelectChat}
//               onNewChat={handleNewChat}
//               onDeleteChat={handleDeleteChat}
//               onLogout={handleLogout}
//             />
//           </div>
//         </div>
//       </div>

//       <div className="main-content">
//         <div className="header">
//           <button className="hamburger-btn" onClick={toggleSidebar}>
//             {sidebarVisible ? '✕' : '☰'}
//           </button>
//           <div className="header-title">MINDORA</div>
//         </div>

//         <div className="chat-content">
//           <div className="chat-panel">
//             <div className="messages">
//               {error && <div className="error-message">{error}</div>}
//               {isLoadingConversations && <div className="loading-text">Loading conversations...</div>}
//               {!isLoadingConversations && !error && !hasStartedChat && messages.length === 0 && <WelcomeScreen />}

//               {messages.map((msg, i) => (
//                 <Message key={i} message={msg} isUser={msg.sender === 'user'} />
//               ))}

//               {loading && <div className="message bot">Mindora is typing...</div>}
//               <div ref={messagesEndRef} />
//             </div>

//             {selectedChat && (
//               <div className="input-bar">
//                 {/* Mic / record controls */}
//                 {!isRecording ? (
//                   <button
//                     className="mic-btn"
//                     title="Record voice"
//                     onClick={startRecording}
//                   >
//                     🎙️
//                   </button>
//                 ) : (
//                   <button
//                     className="mic-btn recording"
//                     title="Stop recording"
//                     onClick={stopRecording}
//                   >
//                     ⏹ {formattedTime}
//                   </button>
//                 )}

//                 <textarea
//                   className="input-box"
//                   value={input}
//                   onChange={e => setInput(e.target.value)}
//                   onKeyDown={handleKeyDown}
//                   placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
//                   rows={1}
//                 />

//                 <button
//                   className="send-btn"
//                   onClick={handleSend}
//                   disabled={loading || !input.trim()}
//                 >
//                   Send
//                 </button>
//               </div>
//             )}

//             {/* Voice preview panel (appears after stopping recording) */}
//             {recordingUrl && (
//               <div className="voice-preview">
//                 <div className="voice-row">
//                   <audio controls src={recordingUrl} />
//                   <span className="duration">{formattedTime}</span>
//                 </div>

//                 <div className="voice-actions">
//                   <button className="secondary" onClick={discardRecording} disabled={uploadingVoice}>
//                     Discard
//                   </button>
//                   <button className="secondary" onClick={insertTranscriptIntoInput} disabled={!sttText}>
//                     Insert Transcript
//                   </button>
//                   <button className="primary" onClick={handleSendVoice} disabled={uploadingVoice}>
//                     {uploadingVoice ? 'Sending…' : 'Send voice'}
//                   </button>
//                 </div>

//                 {!!sttText && (
//                   <div className="stt-box">
//                     <div className="stt-title">Transcript</div>
//                     <div className="stt-text">{sttText}</div>
//                   </div>
//                 )}
//               </div>
//             )}

//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }


// third Iteration
import { useEffect, useState, useRef, useCallback } from 'react';
import { sendMessage, sendVoiceMessage } from '../api/api';
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
      const botMsg = {
        sender: 'bot',
        content: res?.response?.content || 'Sorry, I could not process the audio.',
        timestamp: res?.response?.timestamp || new Date().toISOString(),
        emotion: res?.emotion || null
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

              {messages.map((msg, i) => (
                <Message key={i} message={msg} isUser={msg.sender === 'user'} />
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
