import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage, ChatSession, JobResultData, Job, ContextItem } from '../types';

// Context management settings
const MAX_CONTEXT_TOKENS = 4000;
const CHARS_PER_TOKEN = 4;
const MAX_SESSIONS = 50; // Keep last 50 sessions

// Helper to estimate tokens from text
function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

// Helper to create context-safe summary from job result
function createContextSummary(result?: JobResultData): string {
  if (!result) return '';

  const parts: string[] = [];

  // AI Summary (most important - put first)
  if (result.summary) {
    parts.push(result.summary);
  }

  // Statistics (NDVI, change detection, etc.)
  if (result.statistics) {
    const statsStr = Object.entries(result.statistics.values)
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');
    parts.push(`[${result.statistics.type.toUpperCase()} stats: ${statsStr}]`);
  }

  // Classification (change detection classes)
  if (result.classification) {
    const classStr = Object.entries(result.classification)
      .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v.percentage?.toFixed(1)}%`)
      .join(', ');
    parts.push(`[Classification: ${classStr}]`);
  }

  // Field boundaries
  if (result.fieldBoundaries) {
    parts.push(`[Field boundaries: ${result.fieldBoundaries.numFields} fields detected, total area ${(result.fieldBoundaries.totalAreaM2 / 10000).toFixed(2)} hectares]`);
  }

  // Prithvi reconstruction
  if (result.reconstruction) {
    parts.push(`[Prithvi MAE reconstruction using ${result.reconstruction.model}]`);
  }

  // Include info about visualizations (without the actual images)
  if (result.visualizations && result.visualizations.length > 0) {
    const vizLabels = result.visualizations.map(v => v.label).join(', ');
    parts.push(`[Visualizations available: ${vizLabels}]`);
  }

  return parts.join(' ');
}

// Generate a unique session ID
function generateSessionId(): string {
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

// Generate a session title from the first user message
function generateTitle(messages: ChatMessage[]): string {
  const firstUserMessage = messages.find((m) => m.role === 'user');
  if (firstUserMessage) {
    const content = firstUserMessage.content;
    return content.length > 50 ? content.slice(0, 47) + '...' : content;
  }
  return 'New Chat';
}

interface ChatStore {
  // Current session
  currentSessionId: string | null;
  messages: ChatMessage[];
  activeJobIds: string[];

  // Session history
  sessions: ChatSession[];

  // Context management
  contextTokenCount: number;
  maxContextTokens: number;

  // Job-to-session mapping (for linking from Jobs page)
  jobToSessionMap: Record<string, string>;

  // Actions
  // Session management
  createNewSession: () => string;
  loadSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  getCurrentSession: () => ChatSession | null;
  getSessionByJobId: (jobId: string) => ChatSession | null;

  // Message actions
  addMessage: (message: ChatMessage) => void;
  addJobResultMessage: (job: Job, resultData?: JobResultData) => void;
  updateMessageByJobId: (jobId: string, updates: Partial<ChatMessage>) => void;
  clearMessages: () => void;

  // Job tracking
  trackJob: (jobId: string) => void;
  untrackJob: (jobId: string) => void;

  // Context management
  getContextForLLM: () => ContextItem[];
  getContextForBackend: () => { previous_attachments: Array<{ id: string; filename: string; path: string; mime_type?: string }>, history: string };
  setMaxContextTokens: (tokens: number) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Initial state
      currentSessionId: null,
      messages: [],
      activeJobIds: [],
      sessions: [],
      contextTokenCount: 0,
      maxContextTokens: MAX_CONTEXT_TOKENS,
      jobToSessionMap: {},

      // Session management
      createNewSession: () => {
        const newSessionId = generateSessionId();
        set({
          currentSessionId: newSessionId,
          messages: [],
          activeJobIds: [],
          contextTokenCount: 0,
        });

        return newSessionId;
      },

      loadSession: (sessionId: string) => {
        const state = get();
        const session = state.sessions.find((s) => s.id === sessionId);
        if (session) {
          const tokenCount = session.messages.reduce(
            (acc, m) => acc + estimateTokens(m.content),
            0
          );
          set({
            currentSessionId: sessionId,
            messages: session.messages,
            activeJobIds: [],
            contextTokenCount: tokenCount,
          });
        }
      },

      deleteSession: (sessionId: string) => {
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== sessionId);

          // If deleting current session, create a fresh new session
          if (state.currentSessionId === sessionId) {
            return {
              sessions: newSessions,
              currentSessionId: generateSessionId(),
              messages: [],
              activeJobIds: [],
              contextTokenCount: 0,
            };
          }

          return { sessions: newSessions };
        });
      },

      getCurrentSession: () => {
        const state = get();
        if (!state.currentSessionId) return null;
        return (
          state.sessions.find((s) => s.id === state.currentSessionId) || null
        );
      },

      getSessionByJobId: (jobId: string) => {
        const state = get();
        const sessionId = state.jobToSessionMap[jobId];
        if (!sessionId) return null;
        return state.sessions.find((s) => s.id === sessionId) || null;
      },

      // Message actions
      addMessage: (message) =>
        set((state) => {
          // Create session if none exists
          let sessionId = state.currentSessionId;
          if (!sessionId) {
            sessionId = generateSessionId();
          }

          const newMessage = {
            ...message,
            timestamp: message.timestamp || new Date().toISOString(),
          };
          const newMessages = [...state.messages, newMessage];
          const newTokenCount = newMessages.reduce(
            (acc, m) => acc + estimateTokens(m.content),
            0
          );

          // Update job-to-session mapping
          const jobToSessionMap = { ...state.jobToSessionMap };
          if (message.jobId) {
            jobToSessionMap[message.jobId] = sessionId;
          }

          // Auto-save session to sessions array for visibility in history
          const now = new Date().toISOString();
          const existingIndex = state.sessions.findIndex((s) => s.id === sessionId);
          const updatedSession: ChatSession = {
            id: sessionId,
            title: generateTitle(newMessages),
            messages: newMessages,
            jobIds: newMessages
              .filter((m) => m.jobId)
              .map((m) => m.jobId as string),
            createdAt: existingIndex >= 0 ? state.sessions[existingIndex].createdAt : now,
            updatedAt: now,
          };

          const sessions =
            existingIndex >= 0
              ? state.sessions.map((s, i) => (i === existingIndex ? updatedSession : s))
              : [updatedSession, ...state.sessions];

          return {
            currentSessionId: sessionId,
            messages: newMessages,
            contextTokenCount: newTokenCount,
            jobToSessionMap,
            sessions: sessions.slice(0, MAX_SESSIONS),
          };
        }),

      addJobResultMessage: (job, resultData) =>
        set((state) => {
          const contextSummary = createContextSummary(resultData);
          const message: ChatMessage = {
            role: 'assistant',
            content: contextSummary || `Job ${job.id} completed.`,
            type: 'job_result',
            jobId: job.id,
            timestamp: new Date().toISOString(),
            // Store result data for rendering (includes visualizations)
            resultData: resultData,
          };
          const newMessages = [...state.messages, message];
          const newTokenCount = newMessages.reduce(
            (acc, m) => acc + estimateTokens(m.content),
            0
          );

          // Update job-to-session mapping
          const jobToSessionMap = { ...state.jobToSessionMap };
          const sessionId = state.currentSessionId;
          if (sessionId) {
            jobToSessionMap[job.id] = sessionId;
          }

          // Auto-save session to sessions array
          let sessions = state.sessions;
          if (sessionId) {
            const now = new Date().toISOString();
            const existingIndex = state.sessions.findIndex((s) => s.id === sessionId);
            const updatedSession: ChatSession = {
              id: sessionId,
              title: generateTitle(newMessages),
              messages: newMessages,
              jobIds: newMessages
                .filter((m) => m.jobId)
                .map((m) => m.jobId as string),
              createdAt: existingIndex >= 0 ? state.sessions[existingIndex].createdAt : now,
              updatedAt: now,
            };

            sessions =
              existingIndex >= 0
                ? state.sessions.map((s, i) => (i === existingIndex ? updatedSession : s))
                : [updatedSession, ...state.sessions];
          }

          return {
            messages: newMessages,
            contextTokenCount: newTokenCount,
            activeJobIds: state.activeJobIds.filter((id) => id !== job.id),
            jobToSessionMap,
            sessions: sessions.slice(0, MAX_SESSIONS),
          };
        }),

      updateMessageByJobId: (jobId, updates) =>
        set((state) => ({
          messages: state.messages.map((m) =>
            m.jobId === jobId ? { ...m, ...updates } : m
          ),
        })),

      clearMessages: () =>
        set({ messages: [], contextTokenCount: 0, activeJobIds: [] }),

      // Job tracking
      trackJob: (jobId) =>
        set((state) => {
          // Update job-to-session mapping
          const jobToSessionMap = { ...state.jobToSessionMap };
          if (state.currentSessionId) {
            jobToSessionMap[jobId] = state.currentSessionId;
          }
          return {
            activeJobIds: [...state.activeJobIds, jobId],
            jobToSessionMap,
          };
        }),

      untrackJob: (jobId) =>
        set((state) => ({
          activeJobIds: state.activeJobIds.filter((id) => id !== jobId),
        })),

      // Context management
      getContextForLLM: () => {
        const state = get();
        const maxTokens = state.maxContextTokens;
        const contextItems: ContextItem[] = [];
        let tokenCount = 0;

        const reversedMessages = [...state.messages].reverse();

        for (const message of reversedMessages) {
          const tokens = estimateTokens(message.content);

          if (tokenCount + tokens > maxTokens) {
            break;
          }

          contextItems.unshift({
            role: message.role,
            content: message.content,
            tokenEstimate: tokens,
          });

          tokenCount += tokens;
        }

        return contextItems;
      },

      // Get context for backend job submission (includes previous attachments with paths)
      getContextForBackend: () => {
        const state = get();
        const previousAttachments: Array<{ id: string; filename: string; path: string; mime_type?: string }> = [];
        const seenIds = new Set<string>();

        // Use current messages (includes newly added ones not yet persisted)
        const allMessages = state.messages;

        for (const message of allMessages) {
          if (message.role === 'user' && message.attachments) {
            for (const att of message.attachments) {
              // Check if we have a path and haven't seen this attachment before
              if (att.path && !seenIds.has(att.id)) {
                seenIds.add(att.id);
                previousAttachments.push({
                  id: att.id,
                  filename: att.filename,
                  path: att.path,
                  mime_type: 'image/tiff',
                });
              }
            }
          }
        }

        // Build a compact history summary
        const historyParts: string[] = [];
        for (const message of allMessages.slice(-10)) { // Last 10 messages
          const prefix = message.role === 'user' ? 'User' : 'Assistant';
          const content = message.content.slice(0, 200);
          historyParts.push(`${prefix}: ${content}`);
        }

        return {
          previous_attachments: previousAttachments,
          history: historyParts.join('\n'),
        };
      },

      setMaxContextTokens: (tokens) => set({ maxContextTokens: tokens }),
    }),
    {
      name: 'dt4lc-chat-storage',
      partialize: (state) => {
        // Strip large binary data (base64 images) from messages before persisting
        // This prevents localStorage quota issues with satellite imagery
        const stripBinaryData = (messages: ChatMessage[]): ChatMessage[] => {
          return messages.map((m) => {
            const stripped = { ...m };

            // Remove base64 previews from attachments (keep only metadata)
            if (stripped.attachments) {
              stripped.attachments = stripped.attachments.map((att) => ({
                id: att.id,
                filename: att.filename,
                path: att.path,
                // Omit preview_png_base64 - too large for localStorage
              }));
            }

            // Remove base64 visualizations from result data (can be re-fetched from job)
            if (stripped.resultData?.visualizations) {
              stripped.resultData = {
                ...stripped.resultData,
                visualizations: stripped.resultData.visualizations.map((v) => ({
                  type: v.type,
                  label: v.label,
                  // Omit base64 - too large for localStorage
                  base64: '',
                })),
              };
            }

            return stripped;
          });
        };

        // Strip binary data from both current messages and all sessions
        const strippedSessions = state.sessions.map((session) => ({
          ...session,
          messages: stripBinaryData(session.messages),
        }));

        return {
          sessions: strippedSessions,
          currentSessionId: state.currentSessionId,
          messages: stripBinaryData(state.messages),
          jobToSessionMap: state.jobToSessionMap,
        };
      },
      // Storage version for migrations
      version: 2,
      migrate: (persistedState: unknown, version: number) => {
        // Version 2: Strip binary data from storage to fix quota issues
        if (version < 2) {
          // Clear old storage that might have base64 data
          console.log('[ChatStore] Migrating from v1 to v2: clearing binary data');
          return {
            sessions: [],
            currentSessionId: null,
            messages: [],
            jobToSessionMap: {},
          };
        }
        return persistedState as ChatStore;
      },
      // On rehydration, ensure current session is in the sessions array
      onRehydrateStorage: () => (state) => {
        if (!state) return;

        // If there's a current session with messages, ensure it's in sessions array
        if (state.currentSessionId && state.messages.length > 0) {
          const existingIndex = state.sessions.findIndex(
            (s) => s.id === state.currentSessionId
          );

          if (existingIndex === -1) {
            // Session not in array, add it
            const now = new Date().toISOString();
            const newSession: ChatSession = {
              id: state.currentSessionId,
              title: generateTitle(state.messages),
              messages: state.messages,
              jobIds: state.messages
                .filter((m) => m.jobId)
                .map((m) => m.jobId as string),
              createdAt: now,
              updatedAt: now,
            };
            state.sessions = [newSession, ...state.sessions].slice(0, MAX_SESSIONS);
          } else {
            // Session exists but might have stale messages, update it
            const existingSession = state.sessions[existingIndex];
            const updatedSession: ChatSession = {
              ...existingSession,
              messages: state.messages,
              title: generateTitle(state.messages),
              jobIds: state.messages
                .filter((m) => m.jobId)
                .map((m) => m.jobId as string),
              updatedAt: new Date().toISOString(),
            };
            state.sessions = state.sessions.map((s, i) =>
              i === existingIndex ? updatedSession : s
            );
          }
        }
      },
    }
  )
);
