import React, { useState, useRef, useEffect, useMemo } from "react";
import ReactMarkdown from "react-markdown";

const Avatar = ({ isUser }) => {
  const icon = isUser ? "👤" : "🤖";
  return <div className="avatar">{icon}</div>;
};

const AudioPlayer = ({ src }) => {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      const setAudioData = () => {
        setDuration(audio.duration);
      };

      const setAudioTime = () => setCurrentTime(audio.currentTime);

      const handleEnded = () => setIsPlaying(false);

      audio.addEventListener("loadedmetadata", setAudioData);
      audio.addEventListener("timeupdate", setAudioTime);
      audio.addEventListener("ended", handleEnded);

      return () => {
        audio.removeEventListener("loadedmetadata", setAudioData);
        audio.removeEventListener("timeupdate", setAudioTime);
        audio.removeEventListener("ended", handleEnded);
      };
    }
  }, []);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleProgressClick = (e) => {
    const audio = audioRef.current;
    if (audio && duration) {
      const bounds = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - bounds.left;
      const percentage = x / bounds.width;
      audio.currentTime = percentage * duration;
    }
  };

  const formatTime = (time) => {
    if (time && !isNaN(time)) {
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60);
      return `${minutes}:${seconds.toString().padStart(2, "0")}`;
    }
    return "0:00";
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="audio-player">
      <audio ref={audioRef} src={src} preload="metadata"></audio>
      <button onClick={togglePlayPause} className="play-pause-btn">
        {isPlaying ? "⏸" : "▶"}
      </button>
      <div className="audio-info">
        <div className="progress-bar-container" onClick={handleProgressClick}>
          <div className="progress-bar" style={{ width: `${progress}%` }} />
        </div>
        <div className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>
    </div>
  );
};

// Helper function to remove thinking tags and log them
const removeThinkingTags = (content) => {
  if (!content || typeof content !== "string") {
    return content;
  }

  // Patterns to match various thinking tag formats:
  // - HTML-encoded: &lt;think&gt;, &lt;THINK&gt;, etc.
  // - Regular: <think>, <THINK>, <think >, etc.
  // - With optional whitespace and different capitalizations
  // - Variations: <thinking>, <thought>, etc.

  // Pattern to match think start tags (HTML-encoded and regular, various cases, with optional whitespace)
  // Matches: &lt;think&gt;, &lt;think &gt;, &lt;THINK&gt;, &lt;Think&gt;, &lt;thinking&gt;, <think>, <think >, <THINK>, <Think>, <thinking>, etc.
  const thinkStartStr =
    "(&lt;think\\s*&gt;|&lt;THINK\\s*&gt;|&lt;Think\\s*&gt;|&lt;thinking\\s*&gt;|&lt;THINKING\\s*&gt;|&lt;Thought\\s*&gt;|&lt;THOUGHT\\s*&gt;|<think\\s*>|<THINK\\s*>|<Think\\s*>|<thinking\\s*>|<THINKING\\s*>|<thought\\s*>|<THOUGHT\\s*>)";

  // Pattern to match think end tags
  // Matches: &lt;/think&gt;, &lt;/think &gt;, &lt;/THINK&gt;, &lt;/Think&gt;, &lt;/thinking&gt;, </think>, </think >, </THINK>, </Think>, </thinking>, etc.
  const thinkEndStr =
    "(&lt;\\/think\\s*&gt;|&lt;\\/THINK\\s*&gt;|&lt;\\/Think\\s*&gt;|&lt;\\/thinking\\s*&gt;|&lt;\\/THINKING\\s*&gt;|&lt;\\/Thought\\s*&gt;|&lt;\\/THOUGHT\\s*&gt;|<\\/think\\s*>|<\\/THINK\\s*>|<\\/Think\\s*>|<\\/thinking\\s*>|<\\/THINKING\\s*>|<\\/thought\\s*>|<\\/THOUGHT\\s*>)";

  // Combined pattern to match complete think blocks (start tag, content, end tag)
  const thinkBlockPattern = new RegExp(
    thinkStartStr + "[\\s\\S]*?" + thinkEndStr,
    "gi"
  );

  // Pattern to match standalone think tags (start or end)
  const thinkPattern = new RegExp(thinkStartStr + "|" + thinkEndStr, "gi");

  // Extract and log thinking blocks
  const thinkingBlocks = [];
  let match;
  const pattern = new RegExp(
    thinkStartStr + "([\\s\\S]*?)" + thinkEndStr,
    "gi"
  );

  while ((match = pattern.exec(content)) !== null) {
    const thinkingContent = match[1]; // Content between tags
    if (thinkingContent) {
      thinkingBlocks.push(thinkingContent.trim());
    }
  }

  // Log the thinking parts
  if (thinkingBlocks.length > 0) {
    console.log("Thinking content extracted:", thinkingBlocks);
  }

  // Remove thinking tags from content
  let cleanedContent = content.replace(thinkBlockPattern, "").trim();

  // Also remove any remaining standalone think tags
  cleanedContent = cleanedContent.replace(thinkPattern, "").trim();

  return cleanedContent;
};

// Message Component for rendering individual messages
const Message = ({ message, isUser }) => {
  const [showTranslation, setShowTranslation] = useState(false);

  // Process content to remove thinking tags and log them
  const cleanedContent = useMemo(() => {
    if (!message.content) return message.content;
    return removeThinkingTags(message.content);
  }, [message.content]);

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const renderContent = () => {
    if (message.audioUrl) {
      return (
        <div className="audio-message-content">
          <AudioPlayer src={message.audioUrl} />
          {cleanedContent && (
            <>
              <button
                onClick={() => setShowTranslation(!showTranslation)}
                className="view-translation-btn"
              >
                {showTranslation ? "Hide" : "View"} translation
              </button>
              {showTranslation && (
                <div className="translation-text">
                  <div className="markdown-content">
                    <ReactMarkdown>{cleanedContent}</ReactMarkdown>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      );
    }

    if (isUser) {
      return <div style={{ whiteSpace: "pre-line" }}>{cleanedContent}</div>;
    }

    return (
      <div className="markdown-content">
        <ReactMarkdown
          components={{
            img: ({ node, ...props }) => (
              <img style={{ maxWidth: "100%", height: "auto" }} {...props} />
            ),
            p: ({ children }) => (
              <p style={{ margin: "0 0 12px 0", lineHeight: "1.6" }}>
                {children}
              </p>
            ),
            h1: ({ children }) => (
              <h1
                style={{
                  fontSize: "1.5em",
                  fontWeight: "bold",
                  margin: "0 0 12px 0",
                }}
              >
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2
                style={{
                  fontSize: "1.3em",
                  fontWeight: "bold",
                  margin: "0 0 10px 0",
                }}
              >
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3
                style={{
                  fontSize: "1.2em",
                  fontWeight: "bold",
                  margin: "0 0 8px 0",
                }}
              >
                {children}
              </h3>
            ),
            ul: ({ children }) => (
              <ul style={{ margin: "0 0 12px 0", paddingLeft: "20px" }}>
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol style={{ margin: "0 0 12px 0", paddingLeft: "20px" }}>
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li style={{ margin: "4px 0" }}>{children}</li>
            ),
            blockquote: ({ children }) => (
              <blockquote
                style={{
                  borderLeft: "4px solid currentColor",
                  opacity: "0.7",
                  paddingLeft: "16px",
                  margin: "0 0 12px 0",
                  fontStyle: "italic",
                }}
              >
                {children}
              </blockquote>
            ),
            code: ({ inline, children }) => {
              if (inline) {
                return (
                  <code
                    style={{
                      background: "rgba(0, 0, 0, 0.08)",
                      padding: "2px 6px",
                      borderRadius: "4px",
                      fontSize: "0.9em",
                      fontFamily: "monospace",
                    }}
                  >
                    {children}
                  </code>
                );
              }
              return (
                <code
                  style={{
                    background: "rgba(0, 0, 0, 0.08)",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    fontSize: "0.9em",
                    fontFamily: "monospace",
                    display: "block",
                    overflowX: "auto",
                    maxWidth: "100%",
                  }}
                >
                  {children}
                </code>
              );
            },
            pre: ({ children }) => (
              <pre
                style={{
                  background: "rgba(0, 0, 0, 0.05)",
                  padding: "12px",
                  borderRadius: "8px",
                  overflowX: "auto",
                  margin: "0 0 12px 0",
                  whiteSpace: "pre-wrap",
                  wordWrap: "break-word",
                  maxWidth: "100%",
                }}
              >
                {children}
              </pre>
            ),
            strong: ({ children }) => (
              <strong style={{ fontWeight: "600" }}>{children}</strong>
            ),
            em: ({ children }) => (
              <em style={{ fontStyle: "italic" }}>{children}</em>
            ),
            table: ({ children }) => (
              <table
                style={{
                  borderCollapse: "collapse",
                  width: "100%",
                  margin: "12px 0",
                  display: "block",
                  overflowX: "auto",
                  maxWidth: "100%",
                }}
              >
                {children}
              </table>
            ),
            thead: ({ children }) => (
              <thead style={{ background: "rgba(0, 0, 0, 0.05)" }}>
                {children}
              </thead>
            ),
            tbody: ({ children }) => <tbody>{children}</tbody>,
            tr: ({ children }) => (
              <tr style={{ borderBottom: "1px solid rgba(0, 0, 0, 0.1)" }}>
                {children}
              </tr>
            ),
            th: ({ children }) => (
              <th
                style={{
                  padding: "12px",
                  textAlign: "left",
                  fontWeight: "600",
                  border: "1px solid rgba(0, 0, 0, 0.1)",
                }}
              >
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td
                style={{
                  padding: "12px",
                  border: "1px solid rgba(0, 0, 0, 0.1)",
                }}
              >
                {children}
              </td>
            ),
          }}
        >
          {cleanedContent}
        </ReactMarkdown>
      </div>
    );
  };

  return (
    <div
      className={`message-container ${
        isUser ? "user-container" : "bot-container"
      }`}
    >
      {!isUser && <Avatar isUser={isUser} />}
      <div className="message-content-wrapper">
        <div
          className={`message ${isUser ? "user" : "bot"} ${
            message.audioUrl ? "has-audio" : ""
          }`}
        >
          {renderContent()}
          {!isUser && message.emotion && (
            <div className="emotion-display">
              Emotion detected: <strong>{message.emotion}</strong>
            </div>
          )}
        </div>
        <div className="timestamp">{formatTimestamp(message.timestamp)}</div>
      </div>
      {isUser && <Avatar isUser={isUser} />}
    </div>
  );
};

export default Message;
