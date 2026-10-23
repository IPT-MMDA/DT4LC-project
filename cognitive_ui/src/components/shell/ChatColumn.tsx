import { useChatStore, conversationMeta } from '../../store/useChatStore';
import { attachmentChips } from '../chat/attachmentChips';
import { cn } from '../../utils/cn';
import type { ChatMessage } from '../../types';

// The Workspace's chat column. F1 (shell) lays out the header, the message area
// and the composer box; the message components and the composer controls are
// placeholders that F2 and F3 replace.
export function ChatColumn({ className }: { className?: string }) {
  const conversation = useChatStore((state) =>
    state.conversations.find((c) => c.id === state.activeConversationId),
  );

  return (
    <main
      aria-label="Chat"
      className={cn('flex flex-col min-w-0 min-h-0 bg-bg border-r border-border', className)}
    >
      {conversation && (
        <header className="h-[52px] flex-none flex items-center gap-3 px-5 border-b border-border bg-surface">
          <h1 className="flex-1 min-w-0 truncate text-card">{conversation.title}</h1>
          <span className="text-[11.5px] text-muted font-mono">{conversationMeta(conversation)}</span>
        </header>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto p-5 flex flex-col gap-5">
        {conversation ? <MessagesPlaceholder messages={conversation.messages} /> : <EmptyState />}
      </div>

      <ComposerPlaceholder />
    </main>
  );
}

function EmptyState() {
  return (
    <section className="my-auto px-6 max-w-[472px]" aria-labelledby="empty-title">
      <h2 id="empty-title" className="text-[24px] font-semibold leading-tight mb-4">
        What do you want to know?
      </h2>
      <p className="text-body text-text2">
        Attach a GeoTIFF, pick a region on the map, or start with the Kakhovka sample. Then describe the
        analysis in your own words.
      </p>
    </section>
  );
}

// Plain text per message until F2/F3 bring the message components.
function MessagesPlaceholder({ messages }: { messages: ChatMessage[] }) {
  return (
    <ol className="flex flex-col gap-5">
      {messages.map((m) => (
        <li
          key={m.id}
          className={cn(
            'text-body',
            m.kind === 'user' ? 'self-end max-w-[440px] bg-bubble rounded-composer px-[14px] py-[9px]' : 'self-start',
          )}
        >
          {m.text}
        </li>
      ))}
    </ol>
  );
}

function ComposerPlaceholder() {
  const draft = useChatStore((state) => state.draft);
  const setDraft = useChatStore((state) => state.setDraft);
  const pending = useChatStore((state) => state.pendingAttachments);
  const chips = attachmentChips(pending);

  return (
    <div className="flex-none px-5 pb-4">
      <div className="bg-surface border border-border rounded-composer shadow-composer px-3 py-[10px] flex flex-col gap-2">
        {chips.length > 0 && (
          <ul aria-label="Attachments" className="flex flex-wrap gap-2">
            {chips.map((chip) => (
              <li
                key={chip.key}
                data-testid="attachment-chip"
                className="px-[10px] py-[5px] border border-border rounded-[7px] bg-bg"
              >
                <div className="text-[12px] font-mono">{chip.name}</div>
                {chip.line && <div className="text-[10.5px] text-muted">{chip.line}</div>}
              </li>
            ))}
          </ul>
        )}
        <label className="sr-only" htmlFor="composer-input">
          Message
        </label>
        <textarea
          id="composer-input"
          rows={1}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Describe an analysis... or drop GeoTIFFs here"
          className="resize-none bg-transparent outline-none text-body text-text placeholder:text-faint"
        />
      </div>
    </div>
  );
}
