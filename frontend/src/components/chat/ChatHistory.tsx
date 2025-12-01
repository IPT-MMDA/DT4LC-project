import { MessageSquarePlus, Trash2, MessageSquare } from 'lucide-react';
import { useChatStore } from '../../store/useChatStore';
import type { ChatSession } from '../../types';

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

interface ChatHistoryProps {
  onClose?: () => void;
}

export function ChatHistory({ onClose }: ChatHistoryProps) {
  const {
    sessions,
    currentSessionId,
    createNewSession,
    loadSession,
    deleteSession,
  } = useChatStore();

  const handleNewChat = () => {
    createNewSession();
    onClose?.();
  };

  const handleLoadSession = (sessionId: string) => {
    loadSession(sessionId);
    onClose?.();
  };

  const handleDelete = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (confirm('Delete this chat session?')) {
      deleteSession(sessionId);
    }
  };

  // Sort sessions by updatedAt, newest first
  const sortedSessions = [...sessions].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-950">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
        >
          <MessageSquarePlus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Sessions list */}
      <div className="flex-1 overflow-auto">
        {sortedSessions.length === 0 ? (
          <div className="p-4 text-center text-gray-500 dark:text-gray-400">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No chat history yet</p>
            <p className="text-xs mt-1">Start a new chat to begin</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-800">
            {sortedSessions.map((session: ChatSession) => (
              <div
                key={session.id}
                onClick={() => handleLoadSession(session.id)}
                className={`p-3 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-900 transition-colors group ${
                  session.id === currentSessionId
                    ? 'bg-primary-50 dark:bg-primary-950 border-l-2 border-primary-500'
                    : ''
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {session.title}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {formatDate(session.updatedAt)}
                      </span>
                      {session.jobIds.length > 0 && (
                        <span className="text-xs px-1.5 py-0.5 bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded">
                          {session.jobIds.length} job{session.jobIds.length !== 1 ? 's' : ''}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, session.id)}
                    className="p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Delete chat"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer with stats */}
      {sessions.length > 0 && (
        <div className="p-3 border-t border-gray-200 dark:border-gray-800 text-xs text-gray-500 dark:text-gray-400 text-center">
          {sessions.length} chat{sessions.length !== 1 ? 's' : ''} saved
        </div>
      )}
    </div>
  );
}
