import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  AnalysisEvent,
  AnalysisMessage,
  Attachment,
  ChatMessage,
  ChatRegion,
  Conversation,
  JobResultData,
  NewChatMessage,
  PendingIntent,
  UserMessage,
} from '../types';

// The single owner of chat state: conversations, their messages and the composer.
// Map and UI state stay in useAppStore.

export const CHAT_STORAGE_KEY = 'dt4lc-chat-storage';
export const CHAT_STORAGE_VERSION = 4;
// Retention (design/DESIGN.md §6). Guests: conversations are kept in this browser
// for the session's `kept_hours` after their last update, 24 h until the session is
// known. Members and operators: no time limit. Everyone: at most MAX_CONVERSATIONS,
// the least recently updated dropped first (localStorage holds about 5 MB).
export const GUEST_KEPT_HOURS = 24;
export const MAX_CONVERSATIONS = 50;
export const MAX_CONTEXT_TOKENS = 4000;
const CHARS_PER_TOKEN = 4;
const HOUR_MS = 60 * 60 * 1000;
const NEW_CONVERSATION_TITLE = 'New chat';

const EMPTY_MESSAGES: ChatMessage[] = [];

function generateId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function titleFrom(text: string): string {
  if (!text.trim()) return NEW_CONVERSATION_TITLE;  // files sent without a message
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
        group: att.group,
        mime_type: att.mime_type,
        size_bytes: att.size_bytes,
        metadata: att.metadata,
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

/**
 * The conversations to keep: updated within `keptHours` (null: no time limit), and
 * of those the MAX_CONVERSATIONS most recently updated, in their list order.
 */
export function retainConversations(conversations: Conversation[], keptHours: number | null, now = Date.now()): Conversation[] {
  const cutoff = keptHours === null ? -Infinity : now - keptHours * HOUR_MS;
  const live = conversations.filter((c) => Date.parse(c.updatedAt) >= cutoff);
  if (live.length <= MAX_CONVERSATIONS) return live;
  const newest = [...live].sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)).slice(0, MAX_CONVERSATIONS);
  const keep = new Set(newest.map((c) => c.id));
  return live.filter((c) => keep.has(c.id));
}

export interface ChatStore {
  conversations: Conversation[]; // newest first
  activeConversationId: string | null; // null: the next message starts a conversation
  /** How long this browser keeps conversations: the guest session's `kept_hours`; null for members. */
  keptHours: number | null;

  // Composer (not persisted)
  draft: string;
  pendingAttachments: Attachment[];
  pendingRegion: ChatRegion | null;

  // Conversations
  /** Apply the session's retention (see retainConversations) and drop what it no longer keeps. */
  setKeptHours: (keptHours: number | null) => void;
  newConversation: () => void;
  selectConversation: (conversationId: string) => void;
  renameConversation: (conversationId: string, title: string) => void;
  deleteConversation: (conversationId: string) => void;
  /** Keep the latest reply's `pending` (null clears it); sent back with the next message. */
  setConversationPending: (conversationId: string, pending: PendingIntent | null) => void;

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
  setPendingRegion: (region: ChatRegion | null) => void;
}

/** Messages of the active conversation (a stable empty array when there is none). */
export function selectActiveMessages(state: ChatStore): ChatMessage[] {
  return state.conversations.find((c) => c.id === state.activeConversationId)?.messages ?? EMPTY_MESSAGES;
}

/** The context counter for the active conversation (see estimateTokens). */
export function selectContextTokens(state: ChatStore): number {
  return estimateTokens(selectActiveMessages(state));
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

/**
 * "1 file · 2 results" for the conversation list and the chat header: files are
 * the distinct attachments sent in the conversation (a group such as the sample
 * pair counts once), results the finished analyses that show a result card
 * (`result.card`). Zero parts are left out.
 */
export function conversationMeta(conversation: Conversation): string {
  const files = new Set<string>();
  let results = 0;
  for (const m of conversation.messages) {
    if (m.kind === 'user') m.attachments?.forEach((a) => files.add(a.group?.name ?? a.path ?? a.id));
    if (m.kind === 'analysis' && m.status === 'done' && m.result?.card) results += 1;
  }
  return [files.size > 0 && plural(files.size, 'file'), results > 0 && plural(results, 'result')]
    .filter(Boolean)
    .join(' · ');
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
      keptHours: GUEST_KEPT_HOURS,
      draft: '',
      pendingAttachments: [],
      pendingRegion: null,

      setKeptHours: (keptHours) =>
        set((state) => {
          const conversations = retainConversations(state.conversations, keptHours);
          if (keptHours === state.keptHours && conversations.length === state.conversations.length) return {};
          const active = conversations.some((c) => c.id === state.activeConversationId);
          return { keptHours, conversations, activeConversationId: active ? state.activeConversationId : null };
        }),

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

      setConversationPending: (conversationId, pending) =>
        set((state) => ({
          conversations: state.conversations.map((c) => (c.id === conversationId ? { ...c, pending } : c)),
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
            conversations: retainConversations([conversation, ...state.conversations], state.keptHours),
            activeConversationId: conversation.id,
          });
          return { conversationId: conversation.id, messageId: message.id };
        }

        // The title comes from the first user message with text.
        const isFirstUserMessage =
          message.kind === 'user' &&
          message.text.trim() !== '' &&
          !existing.messages.some((m) => m.kind === 'user' && m.text.trim() !== '');
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

      // Keyed by path: the two files of POST /v1/sample share one id.
      addAttachment: (attachment) =>
        set((state) =>
          state.pendingAttachments.some((a) => a.path === attachment.path)
            ? {}
            : { pendingAttachments: [...state.pendingAttachments, attachment] },
        ),

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
        keptHours: state.keptHours,
      }),
      // v0.5 (version ≤ 3) stored sessions of role/content messages with separate
      // job_submitted / job_result / error entries. That shape does not map onto
      // one analysis message per job, so it is dropped: a v0.5 user starts with an
      // empty conversation list instead of a crash.
      migrate: (persistedState, version) => {
        if (version < CHAT_STORAGE_VERSION) {
          return { conversations: [], activeConversationId: null };
        }
        return persistedState as Pick<ChatStore, 'conversations' | 'activeConversationId' | 'keptHours'>;
      },
      // Retention on load, with the retention of the last known session (a guest's
      // 24 h before any session was seen; v4 data written before F5 has none).
      merge: (persistedState, currentState) => {
        const persisted = (persistedState ?? {}) as Partial<Pick<ChatStore, 'conversations' | 'activeConversationId' | 'keptHours'>>;
        const keptHours = persisted.keptHours === undefined ? GUEST_KEPT_HOURS : persisted.keptHours;
        const conversations = retainConversations(persisted.conversations ?? [], keptHours);
        const activeConversationId = conversations.some((c) => c.id === persisted.activeConversationId)
          ? (persisted.activeConversationId as string)
          : null;
        return { ...currentState, conversations, activeConversationId, keptHours };
      },
    },
  ),
);
