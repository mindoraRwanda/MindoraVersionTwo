// Sidebar Component for chat list management
const Sidebar = ({
  conversations,
  selectedChat,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onLogout
}) => {
  return (
    <div className="sidebar">
      <h3>Chats</h3>
      <button className="new-chat-btn" onClick={onNewChat}>
        + New Chat
      </button>

      {conversations.map(chat => {
        const firstMessage = chat.messages?.find(m => m.sender === 'user')?.content;
        const label = firstMessage
          ? firstMessage.slice(0, 22) + (firstMessage.length > 22 ? '...' : '')
          : `Started ${new Date(chat.started_at).toLocaleDateString()}`;

        return (
          <div key={chat.id} className="chat-item-container">
            <button
              onClick={() => onSelectChat(chat)}
              className={`chat-link ${selectedChat?.id === chat.id ? 'active' : ''}`}
            >
              {label}
            </button>
            <button
              className="delete-btn"
              onClick={(e) => onDeleteChat(chat.id, e)}
              title="Delete this conversation"
            >
              🗑️
            </button>
          </div>
        );
      })}
      <button className="logout-btn" onClick={onLogout}>
        Logout
      </button>
    </div>
  );
};

export default Sidebar;