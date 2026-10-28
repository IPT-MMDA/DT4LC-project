import { useRef, useState, type KeyboardEvent } from 'react';
import { PanelLeftClose, Pencil, Trash2 } from 'lucide-react';
import { useChatStore, conversationMeta } from '../../store/useChatStore';
import type { Conversation } from '../../types';
import { cn } from '../../utils/cn';
import { emptyListText } from './emptyListText';
import { SessionBadge } from './SessionBadge';

const iconButton = 'w-6 h-6 flex items-center justify-center rounded-[5px] text-muted hover:text-text hover:bg-surface';

function ConversationItem({ conversation, active, onPick }: { conversation: Conversation; active: boolean; onPick?: () => void }) {
  const selectConversation = useChatStore((state) => state.selectConversation);
  const renameConversation = useChatStore((state) => state.renameConversation);
  const deleteConversation = useChatStore((state) => state.deleteConversation);
  const [mode, setMode] = useState<'view' | 'rename' | 'confirm-delete'>('view');
  const meta = conversationMeta(conversation);
  const done = useRef(false);  // this rename was committed or cancelled

  const startRename = () => {
    done.current = false;
    setMode('rename');
  };
  const commit = (value: string) => {
    if (done.current) return;  // already done: the blur of the input Enter or Escape removed
    done.current = true;
    const title = value.trim();
    if (title && title !== conversation.title) renameConversation(conversation.id, title);
    setMode('view');
  };
  const onRenameKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') commit(e.currentTarget.value);
    else if (e.key === 'Escape') {
      done.current = true;
      setMode('view');
    }
  };

  return (
    <li className={cn('group relative rounded-control', active ? 'bg-surface2' : 'hover:bg-surface2/60')} data-testid="conversation">
      {mode === 'rename' ? (
        <div className="px-[10px] py-[7px]">
          <input
            aria-label="Conversation name"
            defaultValue={conversation.title}
            autoFocus
            onFocus={(e) => e.currentTarget.select()}
            onKeyDown={onRenameKey}
            onBlur={(e) => commit(e.currentTarget.value)}
            className="w-full bg-surface border border-border rounded-[5px] px-[6px] py-[2px] text-[13px] font-medium focus:border-accent"
          />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => {
            selectConversation(conversation.id);
            onPick?.();
          }}
          aria-current={active ? 'true' : undefined}
          className="w-full text-left px-[10px] py-[9px] rounded-control max-md:pr-[62px] [@media(hover:none)]:pr-[62px]"
        >
          <div className="text-[13px] font-medium leading-[1.3]">{conversation.title}</div>
          {meta && <div className="text-[11px] text-muted mt-[3px]">{meta}</div>}
        </button>
      )}

      {/* Shown on hover and on focus; always on a phone or a touch screen, which have no hover. */}
      {mode === 'view' && (
        <div className="absolute right-[6px] top-[6px] flex gap-[2px] rounded-[6px] bg-surface2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 max-md:opacity-100 [@media(hover:none)]:opacity-100">
          <button type="button" onClick={startRename} aria-label={`Rename “${conversation.title}”`} title="Rename" className={iconButton}>
            <Pencil size={13} aria-hidden />
          </button>
          <button type="button" onClick={() => setMode('confirm-delete')} aria-label={`Delete “${conversation.title}”`} title="Delete" className={iconButton}>
            <Trash2 size={13} aria-hidden />
          </button>
        </div>
      )}

      {mode === 'confirm-delete' && (
        <div role="group" aria-label="Delete this conversation?" className="px-[10px] pb-[9px] flex items-center gap-2 text-[11.5px]">
          <span className="flex-1 text-text2">Delete this conversation?</span>
          <button
            type="button"
            onClick={() => deleteConversation(conversation.id)}
            className="px-2 py-[2px] rounded-[5px] bg-fail text-on-accent font-medium"
          >
            Delete
          </button>
          <button
            type="button"
            autoFocus
            onClick={() => setMode('view')}
            className="px-2 py-[2px] rounded-[5px] border border-border bg-surface text-text2 hover:bg-surface2"
          >
            Cancel
          </button>
        </div>
      )}
    </li>
  );
}

/**
 * The Workspace's left column: conversations (newest first), "+ New", and the session badge.
 * In the phone's menu it fills the drawer (`className`) and `onPick` closes it, the current
 * conversation too; on a tablet `onCollapse` adds a button that hides it.
 */
export function ConversationList({ className, onCollapse, onPick }: { className?: string; onCollapse?: () => void; onPick?: () => void }) {
  const conversations = useChatStore((state) => state.conversations);
  const activeId = useChatStore((state) => state.activeConversationId);
  const keptHours = useChatStore((state) => state.keptHours);
  const newConversation = useChatStore((state) => state.newConversation);

  return (
    <aside
      aria-label="Conversations"
      className={cn('w-[232px] flex-none bg-surface border-r border-border flex flex-col min-h-0', className)}
    >
      <div className="pt-4 px-[14px] pb-[10px] flex items-center justify-between gap-2">
        <h2 className="text-[13px] font-semibold">Conversations</h2>
        {onCollapse && <div className="flex-1" />}
        <button
          type="button"
          onClick={newConversation}
          className="text-[12px] font-medium text-accent-ink bg-accent-tint px-2 py-1 rounded-[5px]"
        >
          + New
        </button>
        {onCollapse && (
          <button
            type="button"
            onClick={onCollapse}
            aria-label="Hide conversations"
            aria-expanded
            title="Hide conversations"
            className="w-7 h-7 flex-none flex items-center justify-center rounded-control text-muted hover:bg-surface2"
          >
            <PanelLeftClose aria-hidden className="w-4 h-4" strokeWidth={1.8} />
          </button>
        )}
      </div>

      {conversations.length === 0 ? (
        <p className="px-5 py-[30px] text-control leading-[1.5] text-muted text-center" data-testid="no-conversations">
          {emptyListText(keptHours)}
        </p>
      ) : (
        <ul className="px-[10px] flex flex-col gap-[2px] overflow-y-auto min-h-0">
          {conversations.map((c) => (
            <ConversationItem key={c.id} conversation={c} active={c.id === activeId} onPick={onPick} />
          ))}
        </ul>
      )}

      <div className="flex-1" />
      <SessionBadge />
    </aside>
  );
}
