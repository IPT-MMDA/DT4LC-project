import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  AnalysisEvent,
  AnalysisMessage,
  Attachment,
  ChatMessage,
  Conversation,
  JobResultData,
  NewChatMessage,
  UserMessage,
} from '../types';
import type { DrawnRegion } from './useAppStore';

// The single owner of chat state: conversations, their messages and the composer.
// Map and UI state stay in useAppStore.

export const CHAT_STORAGE_KEY = 'dt4lc-chat-storage';
export const CHAT_STORAGE_VERSION = 4;
/** Conversations are kept in this browser for 24 h (design/DESIGN.md §6). */
export const CONVERSATION_TTL_MS = 24 * 60 * 60 * 1000;
export const MAX_CONTEXT_TOKENS = 4000;
const CHARS_PER_TOKEN = 4;
const MAX_CONVERSATIONS = 50;
const NEW_CONVERSATION_TITLE = 'New chat';

const EMPTY_MESSAGES: ChatMessage[] = [];

function generateId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function titleFrom(text: string): string {
  return text.length > 50 ? text.slice(0, 47) + '...' : text;
}

// Compact, image-free summary of a job result, for the context counter and the
// history sent to the backend.
function resultSummary(result?: JobResultData): string {
  if (!result) return '';

  const parts: string[] = [];

  if (result.summary) {
    parts.push(result.summary);
  }

  if (result.statistics) {
    const statsStr = Object.entries(result.statistics.values)
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');
    parts.push(`[${result.statistics.type.toUpperCase()} stats: ${statsStr}]`);
  }

  if (result.classification) {
    const classStr = Object.entries(result.classification)
      .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v.percentage?.toFixed(1)}%`)
      .join(', ');
    parts.push(`[Classification: ${classStr}]`);
  }

  if (result.fieldBoundaries) {
    parts.push(`[Field boundaries: ${result.fieldBoundaries.numFields} fields detected, total area ${(result.fieldBoundaries.totalAreaM2 / 10000).toFixed(2)} hectares]`);
  }

  if (result.reconstruction) {
    parts.push(`[Prithvi MAE reconstruction using ${result.reconstruction.model}]`);
  }

  if (result.visualizations && result.visualizations.length > 0) {
    const vizLabels = result.visualizations.map((v) => v.label).join(', ');
    parts.push(`[Visualizations available: ${vizLabels}]`);
  }

  return parts.join(' ');
}

/** The text a message contributes to the conversation context. */
export function messageText(message: ChatMessage): string {
  if (message.kind !== 'analysis') return message.text;
  return [message.text, resultSummary(message.result), message.error].filter(Boolean).join(' ');
}

/**
 * Context counter: an estimate of the tokens in a conversation, at 4 characters
 * per token: ceil(sum of messageText(m).length / 4). User and assistant messages
 * count their text; an analysis counts its narration, its result summary and its
 * error. Image data never reaches messages, so it never counts.
 */
export function estimateTokens(messages: ChatMessage[]): number {
  const chars = messages.reduce((acc, m) => acc + messageText(m).length, 0);
  return Math.ceil(chars / CHARS_PER_TOKEN);
}

/** Context for POST /v1/jobs: earlier attachments with server paths, plus a compact history. */
export function getContextForBackend(messages: ChatMessage[]): {
  previous_attachments: Array<{ id: string; filename: string; path: string; mime_type?: string }>;
  history: string;
} {
  const previousAttachments: Array<{ id: string; filename: string; path: string; mime_type?: string }> = [];
  const seenIds = new Set<string>();

  for (const message of messages) {
    if (message.kind !== 'user' || !message.attachments) continue;
    for (const att of message.attachments) {
      if (att.path && !seenIds.has(att.id)) {
        seenIds.add(att.id);
        previousAttachments.push({ id: att.id, filename: att.filename, path: att.path, mime_type: 'image/tiff' });
      }
    }
  }

  const history = messages
    .slice(-10)
    .map((m) => ({ prefix: m.kind === 'user' ? 'User' : 'Assistant', text: messageText(m) }))
    .filter((m) => m.text)
    .map((m) => `${m.prefix}: ${m.text.slice(0, 200)}`)
    .join('\n');

  return { previous_attachments: previousAttachments, history };
}

// Keep only what localStorage needs to restore a conversation: image URLs, never
// image bytes (#67 bug 3). Attachments keep their metadata and preview URL;
// visualizations keep {type, label, url} and are dropped without a URL; per-tick
// step_progress events are dropped from the timeline.
function toPersistedMessage(message: ChatMessage): ChatMessage {
  if (message.kind === 'user') {
    if (!message.attachments) return message;
    return {
      ...message,
      attachments: message.attachments.map((att) => ({
        id: att.id,
        filename: att.filename,
        path: att.path,
        preview_url: att.preview_url,
      })),
    };
  }

  if (message.kind === 'analysis') {
    const persisted: AnalysisMessage = {
      ...message,
      events: message.events.filter((e) => e.type !== 'step_progress'),
    };
    if (message.result?.visualizations) {
      persisted.result = {
        ...message.result,
        visualizations: message.result.visualizations
          .filter((v) => typeof v.url === 'string' && v.url)
          .map((v) => ({ type: v.type, label: v.label, url: v.url })),
      };
    }
    return persisted;
  }

  return message;
}

export interface ChatStore {
  conversations: Conversation[]; // newest first
  activeConversationId: string | null; // null: the next message starts a conversation

  // Composer (not persisted)
  draft: string;
  pendingAttachments: Attachment[];
  pendingRegion: DrawnRegion | null;

  // Conversations
  newConversation: () => void;
  selectConversation: (conversationId: string) => void;
  renameConversation: (conversationId: string, title: string) => void;
  deleteConversation: (conversationId: string) => void;

  // Messages. addMessage targets `conversationId`, else the active conversation;
  // when that does not exist it starts a new one and makes it active.
  addMessage: (message: NewChatMessage, conversationId?: string) => { conversationId: string; messageId: string };
  updateMessage: (messageId: string, updates: Partial<ChatMessage>) => void;
  removeMessage: (messageId: string) => void;
  // Analyses are found by job id in any conversation: a job may finish after the
  // user has switched conversations.
  updateAnalysis: (jobId: string, updates: Partial<Omit<AnalysisMessage, 'id' | 'kind' | 'jobId'>>) => void;
  appendAnalysisEvent: (jobId: string, event: AnalysisEvent) => void;

  // Composer
  setDraft: (draft: string) => void;
  addAttachment: (attachment: Attachment) => void;
  removeAttachment: (attachmentId: string) => void;
  clearAttachments: () => void;
  setPendingRegion: (region: DrawnRegion | null) => void;
}

/** Messages of the active conversation (a stable empty array when there is none). */
export function selectActiveMessages(state: ChatStore): ChatMessage[] {
  return state.conversations.find((c) => c.id === state.activeConversationId)?.messages ?? EMPTY_MESSAGES;
}

/** The context counter for the active conversation (see estimateTokens). */
export function selectContextTokens(state: ChatStore): number {
  return estimateTokens(selectActiveMessages(state));
}

export function findConversationByJobId(conversations: Conversation[], jobId: string): Conversation | null {
  return (
    conversations.find((c) => c.messages.some((m) => m.kind === 'analysis' && m.jobId === jobId)) ?? null
  );
}

// Apply `update` to the messages of every conversation that `match`es, bumping updatedAt.
function mapMessages(
  conversations: Conversation[],
  match: (m: ChatMessage) => boolean,
  update: (m: ChatMessage) => ChatMessage | null,
): Conversation[] {
  const now = new Date().toISOString();
  return conversations.map((c) => {
    if (!c.messages.some(match)) return c;
    const messages = c.messages.flatMap((m) => {
      if (!match(m)) return [m];
      const next = update(m);
      return next ? [next] : [];
    });
    return { ...c, messages, updatedAt: now };
  });
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeConversationId: null,
      draft: '',
      pendingAttachments: [],
      pendingRegion: null,

      newConversation: () => set({ activeConversationId: null }),

      selectConversation: (conversationId) =>
        set((state) =>
          state.conversations.some((c) => c.id === conversationId) ? { activeConversationId: conversationId } : {},
        ),

      renameConversation: (conversationId, title) =>
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === conversationId ? { ...c, title, updatedAt: new Date().toISOString() } : c,
          ),
        })),

      deleteConversation: (conversationId) =>
        set((state) => ({
          conversations: state.conversations.filter((c) => c.id !== conversationId),
          activeConversationId: state.activeConversationId === conversationId ? null : state.activeConversationId,
        })),

      addMessage: (input, conversationId) => {
        const now = new Date().toISOString();
        const message = { ...input, id: input.id ?? generateId('msg'), createdAt: input.createdAt ?? now } as ChatMessage;
        const state = get();
        const existing = state.conversations.find((c) => c.id === (conversationId ?? state.activeConversationId));

        if (!existing) {
          const conversation: Conversation = {
            id: generateId('chat'),
            title: message.kind === 'user' ? titleFrom(message.text) : NEW_CONVERSATION_TITLE,
            createdAt: now,
            updatedAt: now,
            messages: [message],
          };
          set({
            conversations: [conversation, ...state.conversations].slice(0, MAX_CONVERSATIONS),
            activeConversationId: conversation.id,
          });
          return { conversationId: conversation.id, messageId: message.id };
        }

        const isFirstUserMessage = message.kind === 'user' && !existing.messages.some((m) => m.kind === 'user');
        const updated: Conversation = {
          ...existing,
          title: isFirstUserMessage ? titleFrom((message as UserMessage).text) : existing.title,
          messages: [...existing.messages, message],
          updatedAt: now,
        };
        set({ conversations: state.conversations.map((c) => (c.id === existing.id ? updated : c)) });
        return { conversationId: existing.id, messageId: message.id };
      },

      updateMessage: (messageId, updates) =>
        set((state) => ({
          conversations: mapMessages(
            state.conversations,
            (m) => m.id === messageId,
            (m) => ({ ...m, ...updates, id: m.id, kind: m.kind }) as ChatMessage,
          ),
        })),

      removeMessage: (messageId) =>
        set((state) => ({
          conversations: mapMessages(state.conversations, (m) => m.id === messageId, () => null),
        })),

      updateAnalysis: (jobId, updates) =>
        set((state) => ({
          conversations: mapMessages(
            state.conversations,
            (m) => m.kind === 'analysis' && m.jobId === jobId,
            (m) => ({ ...(m as AnalysisMessage), ...updates }),
          ),
        })),

      appendAnalysisEvent: (jobId, event) =>
        set((state) => ({
          conversations: mapMessages(
            state.conversations,
            (m) => m.kind === 'analysis' && m.jobId === jobId,
            (m) => ({ ...(m as AnalysisMessage), events: [...(m as AnalysisMessage).events, event] }),
          ),
        })),

      setDraft: (draft) => set({ draft }),

      addAttachment: (attachment) =>
        set((state) => ({ pendingAttachments: [...state.pendingAttachments, attachment] })),

      removeAttachment: (attachmentId) =>
        set((state) => ({ pendingAttachments: state.pendingAttachments.filter((a) => a.id !== attachmentId) })),

      clearAttachments: () => set({ pendingAttachments: [] }),

      setPendingRegion: (region) => set({ pendingRegion: region }),
    }),
    {
      name: CHAT_STORAGE_KEY,
      version: CHAT_STORAGE_VERSION,
      // Only what restores a conversation; the composer starts empty on every load.
      partialize: (state) => ({
        conversations: state.conversations.map((c) => ({ ...c, messages: c.messages.map(toPersistedMessage) })),
        activeConversationId: state.activeConversationId,
      }),
      // v0.5 (version ≤ 3) stored sessions of role/content messages with separate
      // job_submitted / job_result / error entries. That shape does not map onto
      // one analysis message per job, so it is dropped: a v0.5 user starts with an
      // empty conversation list instead of a crash.
      migrate: (persistedState, version) => {
        if (version < CHAT_STORAGE_VERSION) {
          return { conversations: [], activeConversationId: null };
        }
        return persistedState as Pick<ChatStore, 'conversations' | 'activeConversationId'>;
      },
      // 24 h retention: drop conversations not updated within the TTL on load.
      merge: (persistedState, currentState) => {
        const persisted = (persistedState ?? {}) as Partial<Pick<ChatStore, 'conversations' | 'activeConversationId'>>;
        const cutoff = Date.now() - CONVERSATION_TTL_MS;
        const conversations = (persisted.conversations ?? []).filter((c) => Date.parse(c.updatedAt) >= cutoff);
        const activeConversationId = conversations.some((c) => c.id === persisted.activeConversationId)
          ? (persisted.activeConversationId as string)
          : null;
        return { ...currentState, conversations, activeConversationId };
      },
    },
  ),
);
