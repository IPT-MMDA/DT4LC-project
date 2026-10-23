import { useCallback, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ChatError, streamChat, toChatAttachment, toChatRegion, type ChatTurn } from '../api/chat';
import { SESSION_QUERY_KEY } from '../api/hooks/useSession';
import { messageText, useChatStore } from '../store/useChatStore';
import type { AnalysisEvent, AnalysisMessage, Attachment, ChatMessage, ChatRegion, MessageAttachment } from '../types';

// One Workspace message through POST /v1/chat (SSE): the user message, then a
// streamed markdown reply or one analysis message. CONTRACTS "Conversation path".

const HISTORY_TURNS = 10;  // the server keeps the last 10 turns of at most 2000 chars
const TURN_CHARS = 2000;

/** `data` of a stored job event ({seq, type, data}), or the event itself when flattened. */
function eventData(e: AnalysisEvent): Record<string, unknown> {
  return typeof e.data === 'object' && e.data !== null ? (e.data as Record<string, unknown>) : e;
}

/** The intent sentence of an analysis: its narration, else its `intent` event. */
export function analysisIntent(m: AnalysisMessage): string {
  if (m.text) return m.text;
  const intent = m.events.find((e) => e.type === 'intent');
  return intent ? String(eventData(intent).text ?? '') : '';
}

/** One line for `history`: what was asked and how it ended; '' while it runs. */
export function analysisSummary(m: AnalysisMessage): string {
  const last = [...m.events].reverse().find((e) => ['done', 'failed', 'cancelled'].includes(e.type));
  const data = last ? eventData(last) : {};
  let outcome = '';
  if (m.status === 'done' || last?.type === 'done') {
    outcome = ['Done', data.summary, m.result?.summary].filter(Boolean).join(': ');
  } else if (m.status === 'failed' || last?.type === 'failed') {
    outcome = `Failed: ${String(data.message ?? m.error ?? 'error')}`;
  } else if (m.status === 'cancelled' || last?.type === 'cancelled') {
    outcome = 'Cancelled';
  }
  if (!outcome) return '';
  const intent = analysisIntent(m) || messageText(m);
  return `${[intent, outcome].filter(Boolean).join(' ')}`.replace(/\s+/g, ' ').trim();
}

/** `history` for POST /v1/chat: the last turns, each finished analysis as one assistant line. */
export function conversationHistory(messages: ChatMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (const m of messages) {
    if (m.kind === 'user' && m.text.trim()) turns.push({ role: 'user', text: m.text });
    else if (m.kind === 'assistant' && !m.error && m.text.trim()) turns.push({ role: 'assistant', text: m.text });
    else if (m.kind === 'analysis') {
      const line = analysisSummary(m);
      if (line) turns.push({ role: 'assistant', text: line });
    }
  }
  return turns.slice(-HISTORY_TURNS).map((t) => ({ ...t, text: t.text.slice(0, TURN_CHARS) }));
}

/**
 * The conversation's current files or region: those of the latest user message
 * that had any. A follow-up re-sends them, since /v1/chat keeps no attachments;
 * attaching new files or a region replaces them.
 */
export function carriedContext(messages: ChatMessage[]): { attachments: MessageAttachment[]; region?: ChatRegion } {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.kind !== 'user') continue;
    if (m.attachments?.length || m.region) return { attachments: m.attachments ?? [], region: m.region };
  }
  return { attachments: [] };
}

function toMessageAttachment(a: Attachment): MessageAttachment {
  return {
    id: a.id,
    filename: a.filename,
    path: a.path,
    preview_url: a.preview_url,
    group: a.group,
    mime_type: a.mime_type,
    size_bytes: a.size_bytes,
    metadata: a.metadata,
  };
}

export interface SendOptions {
  /** After an `analysis` frame, or a quota refusal: the session's quota changed. */
  onQuotaChanged?: () => void;
  /** A `retry` fix's `params`, merged into the request body (docs/API.md "Failures"). */
  params?: Record<string, unknown>;
}

/**
 * Send `text` with the composer's files and region (or the conversation's
 * current ones), its history and its `pending`. Clears the composer. Never throws:
 * a refusal or a network error becomes an inline assistant error.
 */
export async function sendMessage(text: string, options: SendOptions = {}): Promise<void> {
  const store = useChatStore.getState();
  const message = text.trim();
  const composerFiles = store.pendingAttachments;
  const composerRegion = store.pendingRegion;
  if (!message && composerFiles.length === 0 && !composerRegion) return;

  const conversation = store.conversations.find((c) => c.id === store.activeConversationId);
  const earlier = conversation?.messages ?? [];
  const fresh = composerFiles.length > 0 || composerRegion !== null;
  const context = fresh
    ? { attachments: composerFiles.map(toMessageAttachment), region: composerRegion ?? undefined }
    : carriedContext(earlier);

  const { conversationId, messageId: userMessageId } = store.addMessage({
    kind: 'user',
    text: message,
    ...(fresh && context.attachments.length > 0 ? { attachments: context.attachments } : {}),
    ...(fresh && context.region ? { region: context.region } : {}),
  });
  store.setDraft('');
  store.clearAttachments();
  store.setPendingRegion(null);

  const { addMessage, updateMessage, setConversationPending } = useChatStore.getState();
  let replyId: string | null = null;
  let reply = '';
  const showReply = (value: string) => {
    reply = value;
    if (replyId) updateMessage(replyId, { text: reply });
    else replyId = addMessage({ kind: 'assistant', text: reply }, conversationId).messageId;
  };

  try {
    await streamChat(
      {
        message,
        history: conversationHistory(earlier),
        attachments: context.attachments.filter((a) => a.path).map((a) => toChatAttachment(a as Attachment)),
        ...(context.region ? { region: toChatRegion(context.region) } : {}),
        ...(conversation?.pending ? { pending: conversation.pending } : {}),
        ...options.params,
      },
      (event) => {
        if (event.type === 'reply_delta') {
          showReply(reply + event.text);
        } else if (event.type === 'reply_done') {
          showReply(event.text);
          setConversationPending(conversationId, event.pending ?? null);
        } else if (event.type === 'analysis') {
          setConversationPending(conversationId, null);
          addMessage(
            { kind: 'analysis', jobId: event.job_id, status: 'pending', text: '', events: event.job.events ?? [] },
            conversationId,
          );
          const echo = event.job.region;
          if (fresh && context.region && echo?.line) {
            updateMessage(userMessageId, {
              region: { ...context.region, line: echo.line, area_km2: echo.area_km2 ?? context.region.area_km2 },
            });
          }
          options.onQuotaChanged?.();
        }
      },
    );
  } catch (err) {
    const quota = err instanceof ChatError && err.code === 'GUEST_QUOTA';
    const reason = err instanceof Error ? err.message : String(err);
    addMessage({ kind: 'assistant', text: quota ? reason : `Couldn't send the message: ${reason}`, error: true }, conversationId);
    if (quota) options.onQuotaChanged?.();
  }
}

/** `send(text?, {params}?)` (the draft by default) and whether a message is in flight. */
export function useSendMessage() {
  const queryClient = useQueryClient();
  const [sending, setSending] = useState(false);
  const inFlight = useRef(false);

  const send = useCallback(
    async (text?: string, options: Pick<SendOptions, 'params'> = {}) => {
      if (inFlight.current) return;
      inFlight.current = true;
      setSending(true);
      try {
        await sendMessage(text ?? useChatStore.getState().draft, {
          ...options,
          onQuotaChanged: () => queryClient.invalidateQueries({ queryKey: SESSION_QUERY_KEY }),
        });
      } finally {
        inFlight.current = false;
        setSending(false);
      }
    },
    [queryClient],
  );

  return { send, sending };
}
