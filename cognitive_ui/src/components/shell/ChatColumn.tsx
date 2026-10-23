import { useEffect, useRef, useState, type DragEvent } from 'react';
import { useChatStore, conversationMeta } from '../../store/useChatStore';
import { useAttachFiles } from '../../hooks/useAttachFiles';
import { useSendMessage } from '../../hooks/useSendMessage';
import { Composer } from '../chat/Composer';
import { EmptyState } from '../chat/EmptyState';
import { MessageList } from '../chat/MessageList';
import { cn } from '../../utils/cn';

const hasFiles = (e: DragEvent) => e.dataTransfer?.types?.includes('Files') ?? false;

// The Workspace's chat column: header, messages (or the empty state) and the
// composer. The whole column is a drop zone for GeoTIFFs.
export function ChatColumn({ className }: { className?: string }) {
  const conversation = useChatStore((state) =>
    state.conversations.find((c) => c.id === state.activeConversationId),
  );
  const messages = conversation?.messages ?? [];
  const { send, sending } = useSendMessage();
  const { attachFiles, uploading, error } = useAttachFiles();
  const fileInput = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const openFilePicker = () => fileInput.current?.click();

  // Follow the conversation: new messages and streamed reply text.
  const last = messages[messages.length - 1];
  const lastText = last?.text;
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, lastText, conversation?.id]);

  const onDragOver = (e: DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    if (!dragging) setDragging(true);
  };
  const onDragLeave = (e: DragEvent) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false);
  };
  const onDrop = (e: DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    setDragging(false);
    void attachFiles(e.dataTransfer.files);
  };

  return (
    <main
      aria-label="Chat"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={cn('relative flex flex-col min-w-0 min-h-0 bg-bg border-r border-border', className)}
    >
      {conversation && (
        <header className="h-[52px] flex-none flex items-center gap-3 px-5 border-b border-border bg-surface">
          <h1 className="flex-1 min-w-0 truncate text-card">{conversation.title}</h1>
          <span className="text-[11.5px] text-muted font-mono">{conversationMeta(conversation)}</span>
        </header>
      )}

      <div ref={scroller} className="flex-1 min-h-0 overflow-y-auto p-5 flex flex-col gap-5">
        {messages.length > 0 ? (
          <MessageList messages={messages} waiting={sending} />
        ) : (
          <EmptyState onSend={(prompt) => void send(prompt)} onOpenFilePicker={openFilePicker} sending={sending} />
        )}
      </div>

      <Composer
        hasMessages={messages.length > 0}
        sending={sending}
        onSend={() => void send()}
        onOpenFilePicker={openFilePicker}
        note={uploading ? 'Uploading…' : error}
      />

      <input
        ref={fileInput}
        type="file"
        accept=".tif,.tiff,image/tiff"
        multiple
        hidden
        data-testid="file-input"
        onChange={(e) => {
          if (e.target.files) void attachFiles(e.target.files);
          e.target.value = '';
        }}
      />

      {dragging && (
        <div
          aria-hidden
          className="absolute inset-2 z-30 rounded-card border-2 border-dashed border-accent bg-accent-tint/80 flex items-center justify-center text-card text-accent-ink pointer-events-none"
        >
          Drop GeoTIFFs to attach
        </div>
      )}
    </main>
  );
}
