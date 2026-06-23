import React from 'react';

const getDateLabel = (dateStr) => {
  const date = new Date(dateStr);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);

  const sameDay = (a, b) =>
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate();

  if (sameDay(date, today)) return 'Today';
  if (sameDay(date, yesterday)) return 'Yesterday';

  return date.toLocaleDateString([], { year: 'numeric', month: 'long', day: 'numeric' });
};

const groupByDate = (conversations) => {
  const groups = [];
  const seen = new Map(); // label → group index

  for (const chat of conversations) {
    const label = getDateLabel(chat.started_at);
    if (seen.has(label)) {
      groups[seen.get(label)].chats.push(chat);
    } else {
      seen.set(label, groups.length);
      groups.push({ label, chats: [chat] });
    }
  }

  return groups;
};

const Sidebar = ({
  conversations,
  selectedChat,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onLogout
}) => {
  const groups = groupByDate(conversations);

  return (
    <>
      <h3 className="sidebar-heading">Chats</h3>
      <button className="new-chat-btn" onClick={onNewChat}>
        + New Chat
      </button>

      <div className="chat-list">
        {groups.map(({ label, chats }) => (
          <div key={label} className="chat-date-group">
            <p className="chat-date-label">{label}</p>

            {chats.map(chat => {
              const firstMessage = chat.messages?.find(m => m.sender === 'user')?.content;
              const chatLabel = chat.title
                ? chat.title
                : firstMessage
                  ? firstMessage.slice(0, 22) + (firstMessage.length > 22 ? '...' : '')
                  : 'New conversation';

              return (
                <div key={chat.id} className="chat-item-container">
                  <button
                    onClick={() => onSelectChat(chat)}
                    className={`chat-link ${selectedChat?.id === chat.id ? 'active' : ''}`}
                  >
                    {chatLabel}
                  </button>
                  <button
                    className="delete-btn"
                    onClick={(e) => onDeleteChat(chat.id, e)}
                    title="Delete this conversation"
                    aria-label="Delete conversation"
                  >
                    🗑️
                  </button>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <button className="logout-btn" onClick={onLogout}>
        Logout
      </button>
    </>
  );
};

export default Sidebar;
