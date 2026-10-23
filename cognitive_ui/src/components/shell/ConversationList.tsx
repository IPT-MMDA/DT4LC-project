import { useChatStore, conversationMeta } from '../../store/useChatStore';
import { cn } from '../../utils/cn';
import { SessionBadge } from './SessionBadge';

/** The Workspace's left column: conversations (newest first), "+ New", and the session badge. */
export function ConversationList() {
  const conversations = useChatStore((state) => state.conversations);
  const activeId = useChatStore((state) => state.activeConversationId);
  const selectConversation = useChatStore((state) => state.selectConversation);
  const newConversation = useChatStore((state) => state.newConversation);

  return (
    <aside
      aria-label="Conversations"
      className="w-[232px] flex-none bg-surface border-r border-border flex flex-col min-h-0"
    >
      <div className="pt-4 px-[14px] pb-[10px] flex items-center justify-between">
        <h2 className="text-[13px] font-semibold">Conversations</h2>
        <button
          type="button"
          onClick={newConversation}
          className="text-[12px] font-medium text-accent-ink bg-accent-tint px-2 py-1 rounded-[5px]"
        >
          + New
        </button>
      </div>

      {conversations.length === 0 ? (
        <p className="px-5 py-[30px] text-control leading-[1.5] text-faint text-center">
          No conversations yet. They appear here as you work and stay for 24 h.
        </p>
      ) : (
        <ul className="px-[10px] flex flex-col gap-[2px] overflow-y-auto min-h-0">
          {conversations.map((c) => {
            const active = c.id === activeId;
            const meta = conversationMeta(c);
            return (
              <li key={c.id}>
                <button
                  type="button"
                  onClick={() => selectConversation(c.id)}
                  aria-current={active ? 'true' : undefined}
                  className={cn(
                    'w-full text-left px-[10px] py-[9px] rounded-control',
                    active ? 'bg-surface2' : 'hover:bg-surface2/60',
                  )}
                >
                  <div className="text-[13px] font-medium leading-[1.3]">{c.title}</div>
                  {meta && <div className="text-[11px] text-muted mt-[3px]">{meta}</div>}
                </button>
              </li>
            );
          })}
        </ul>
      )}

      <div className="flex-1" />
      <SessionBadge />
    </aside>
  );
}
