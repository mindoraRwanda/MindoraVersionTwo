import ReactMarkdown from "react-markdown";
import { useMemo } from "react";

// Message Component for rendering individual messages
const Message = ({ message, isUser }) => {
  // Remove thinking tags and detect if thinking content existed
  const { cleanedContent, hasThinking } = useMemo(() => {
    if (isUser || !message.content) {
      return { cleanedContent: message.content || "", hasThinking: false };
    }

    // Check if content contains thinking tags (handle both HTML-encoded and regular tags)
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

    const hasThinking = thinkBlockPattern.test(message.content);

    // Remove thinking tags from content
    let cleanedContent = message.content.replace(thinkBlockPattern, "").trim();

    // Also remove any remaining standalone think tags
    cleanedContent = cleanedContent.replace(thinkPattern, "").trim();

    return { cleanedContent, hasThinking };
  }, [message.content, isUser]);
  console.log("Thinking", hasThinking, message);

  const renderContent = () => {
    if (isUser) {
      return <div style={{ whiteSpace: "pre-line" }}>{message.content}</div>;
    }

    return (
      <div className="markdown-content">
        <ReactMarkdown
          components={{
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
                  color: "#1f2937",
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
                  color: "#374151",
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
                  color: "#4b5563",
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
                  borderLeft: "4px solid #e5e7eb",
                  paddingLeft: "16px",
                  margin: "0 0 12px 0",
                  fontStyle: "italic",
                  color: "#6b7280",
                }}
              >
                {children}
              </blockquote>
            ),
            code: ({ children }) => (
              <code
                style={{
                  background: "#f3f4f6",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  fontSize: "0.9em",
                  fontFamily: "monospace",
                }}
              >
                {children}
              </code>
            ),
            pre: ({ children }) => (
              <pre
                style={{
                  background: "#1f2937",
                  color: "#f9fafb",
                  padding: "12px",
                  borderRadius: "8px",
                  overflowX: "auto",
                  margin: "0 0 12px 0",
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
                }}
              >
                {children}
              </table>
            ),
            thead: ({ children }) => (
              <thead style={{ background: "#f3f4f6" }}>{children}</thead>
            ),
            tbody: ({ children }) => <tbody>{children}</tbody>,
            tr: ({ children }) => (
              <tr style={{ borderBottom: "1px solid #e5e7eb" }}>{children}</tr>
            ),
            th: ({ children }) => (
              <th
                style={{
                  padding: "12px",
                  textAlign: "left",
                  fontWeight: "600",
                  border: "1px solid #d1d5db",
                }}
              >
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td style={{ padding: "12px", border: "1px solid #d1d5db" }}>
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
    <div className={`message ${isUser ? "user" : "bot"}`}>
      {renderContent()}
      {!isUser && message.emotion && (
        <div style={{ fontSize: "12px", marginTop: "6px", color: "#6b7280" }}>
          Emotion detected: <strong>{message.emotion}</strong>
        </div>
      )}
    </div>
  );
};

export default Message;
