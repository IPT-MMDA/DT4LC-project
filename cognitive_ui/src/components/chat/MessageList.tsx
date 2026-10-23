import { analysisIntent } from '../../hooks/useSendMessage';
import type { AnalysisMessage, AssistantMessage, ChatMessage, UserMessage } from '../../types';
import { attachmentChips } from './attachmentChips';
import { FileChip, RegionChip } from './Chips';
import { Markdown } from './Markdown';

/** The DT monogram beside every assistant message. */
export function Avatar() {
  return (
    <div
      aria-hidden
      className="w-6 h-6 mt-[1px] flex-none rounded-[6px] bg-ink text-ink-bg font-bold text-[9px] flex items-center justify-center"
    >
      DT
    </div>
  );
}

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

// F3 replaces this with the analysis message (timeline, result, fixes).
function AnalysisPlaceholder({ message }: { message: AnalysisMessage }) {
  return (
    <li className="flex gap-3 min-w-0" data-testid="analysis-message" data-job-id={message.jobId}>
      <Avatar />
      <div className="flex-1 min-w-0 text-body">
        <p>{analysisIntent(message) || 'Starting the analysis…'}</p>
        <p className="text-control text-muted mt-1">Analysis {message.status}</p>
      </div>
    </li>
  );
}

export function MessageList({ messages, waiting }: { messages: ChatMessage[]; waiting: boolean }) {
  const last = messages[messages.length - 1];
  return (
    <ol className="flex flex-col gap-5" aria-label="Messages" aria-busy={waiting}>
      {messages.map((m) =>
        m.kind === 'user' ? (
          <UserBubble key={m.id} message={m} />
        ) : m.kind === 'assistant' ? (
          <AssistantText key={m.id} message={m} />
        ) : (
          <AnalysisPlaceholder key={m.id} message={m} />
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
