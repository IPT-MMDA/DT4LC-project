import { carriedContext } from '../../hooks/useSendMessage';
import type { AssistantMessage, ChatMessage, UserMessage } from '../../types';
import { AnalysisMessage, type AnalysisMessageProps } from './analysis/AnalysisMessage';
import { attachmentChips } from './attachmentChips';
import { Avatar } from './Avatar';
import { FileChip, RegionChip } from './Chips';
import { Markdown } from './Markdown';

function UserBubble({ message }: { message: UserMessage }) {
  const chips = attachmentChips(message.attachments ?? []);
  return (
    <li className="self-end max-w-[440px] flex flex-col items-end gap-2" data-testid="user-message">
      {chips.map((chip) => (
        <FileChip key={chip.key} chip={chip} />
      ))}
      {message.region && <RegionChip region={message.region} />}
      {message.text && (
        <div className="bg-bubble rounded-composer px-[14px] py-[9px] text-body whitespace-pre-wrap break-words">
          {message.text}
        </div>
      )}
    </li>
  );
}

function AssistantText({ message }: { message: AssistantMessage }) {
  return (
    <li className="flex gap-3 min-w-0" data-testid="assistant-message">
      <Avatar />
      <div className="flex-1 min-w-0">
        {message.error ? (
          <p role="alert" className="text-body text-fail-ink">
            {message.text}
          </p>
        ) : (
          <Markdown text={message.text} />
        )}
      </div>
    </li>
  );
}

/** The text of the user message before `index`: what an analysis at `index` was asked. */
function promptBefore(messages: ChatMessage[], index: number): string {
  for (let i = index - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.kind === 'user') return m.text;
  }
  return '';
}

interface MessageListProps extends Pick<AnalysisMessageProps, 'onSend' | 'onOpenFilePicker'> {
  messages: ChatMessage[];
  waiting: boolean;
}

export function MessageList({ messages, waiting, onSend, onOpenFilePicker }: MessageListProps) {
  const last = messages[messages.length - 1];
  return (
    <ol className="flex flex-col gap-5" aria-label="Messages" aria-busy={waiting}>
      {messages.map((m, i) =>
        m.kind === 'user' ? (
          <UserBubble key={m.id} message={m} />
        ) : m.kind === 'assistant' ? (
          <AssistantText key={m.id} message={m} />
        ) : (
          <AnalysisMessage
            key={m.id}
            message={m}
            prompt={promptBefore(messages, i)}
            context={carriedContext(messages.slice(0, i))}
            onSend={onSend}
            onOpenFilePicker={onOpenFilePicker}
          />
        ),
      )}
      {waiting && last?.kind === 'user' && (
        <li className="flex gap-3" aria-live="polite">
          <Avatar />
          <span className="text-body text-muted">…</span>
        </li>
      )}
    </ol>
  );
}
