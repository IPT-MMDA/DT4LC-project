import { create } from 'zustand';
import type { ChatMessage, Attachment, ContextItem, JobResultData, Job } from '../types';

interface Layer {
  id: string;
  name: string;
  type: string;
  visible: boolean;
  opacity: number;
}

// Context management settings
const MAX_CONTEXT_TOKENS = 4000; // Safe limit for small LLM instances
const CHARS_PER_TOKEN = 4; // Rough estimate

interface AppStore {
  // UI State
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark' | 'auto';

  // Map State
  currentBasemap: string;
  mapLayers: Layer[];
  selectedFeature: any | null;

  // Chat State
  messages: ChatMessage[];
  uploadedAttachments: Attachment[];
  activeJobIds: string[]; // Jobs being tracked in chat

  // Context Management
  contextTokenCount: number;
  maxContextTokens: number;

  // Actions
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;

  // Map Actions
  setBasemap: (basemap: string) => void;
  addLayer: (layer: Layer) => void;
  removeLayer: (layerId: string) => void;
  toggleLayerVisibility: (layerId: string) => void;
  setLayerOpacity: (layerId: string, opacity: number) => void;
  setSelectedFeature: (feature: any | null) => void;

  // Chat Actions
  addMessage: (message: ChatMessage) => void;
  addJobResultMessage: (job: Job, resultData?: JobResultData) => void;
  updateMessageByJobId: (jobId: string, updates: Partial<ChatMessage>) => void;
  clearMessages: () => void;

  // Attachment Actions
  addAttachment: (attachment: Attachment) => void;
  removeAttachment: (attachmentId: string) => void;
  clearAttachments: () => void;

  // Job Tracking
  trackJob: (jobId: string) => void;
  untrackJob: (jobId: string) => void;

  // Context Management
  getContextForLLM: () => ContextItem[];
  setMaxContextTokens: (tokens: number) => void;
}

// Helper to estimate tokens from text
function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

// Helper to create context-safe summary from job result
function createContextSummary(result?: JobResultData): string {
  if (!result) return '';

  const parts: string[] = [];

  if (result.summary) {
    parts.push(result.summary);
  }

  if (result.statistics) {
    const statsStr = Object.entries(result.statistics.values)
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');
    parts.push(`[${result.statistics.type} stats: ${statsStr}]`);
  }

  if (result.fieldBoundaries) {
    parts.push(`[Field boundaries: ${result.fieldBoundaries.numFields} fields, ${(result.fieldBoundaries.totalAreaM2 / 10000).toFixed(2)} hectares]`);
  }

  if (result.features) {
    parts.push(`[Prithvi features: ${result.features.dimensions}D vector]`);
  }

  return parts.join(' ');
}

export const useAppStore = create<AppStore>((set, get) => ({
  // Initial state
  sidebarCollapsed: false,
  theme: 'light',
  currentBasemap: 'streets',
  mapLayers: [],
  selectedFeature: null,
  messages: [],
  uploadedAttachments: [],
  activeJobIds: [],
  contextTokenCount: 0,
  maxContextTokens: MAX_CONTEXT_TOKENS,

  // UI Actions
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

  setSidebarCollapsed: (collapsed) =>
    set({ sidebarCollapsed: collapsed }),

  setTheme: (theme) =>
    set({ theme }),

  // Map Actions
  setBasemap: (basemap) =>
    set({ currentBasemap: basemap }),

  addLayer: (layer) =>
    set((state) => ({ mapLayers: [...state.mapLayers, layer] })),

  removeLayer: (layerId) =>
    set((state) => ({
      mapLayers: state.mapLayers.filter((l) => l.id !== layerId),
    })),

  toggleLayerVisibility: (layerId) =>
    set((state) => ({
      mapLayers: state.mapLayers.map((l) =>
        l.id === layerId ? { ...l, visible: !l.visible } : l
      ),
    })),

  setLayerOpacity: (layerId, opacity) =>
    set((state) => ({
      mapLayers: state.mapLayers.map((l) =>
        l.id === layerId ? { ...l, opacity } : l
      ),
    })),

  setSelectedFeature: (feature) =>
    set({ selectedFeature: feature }),

  // Chat Actions
  addMessage: (message) =>
    set((state) => {
      const newMessage = {
        ...message,
        timestamp: message.timestamp || new Date().toISOString(),
      };
      const newMessages = [...state.messages, newMessage];
      const newTokenCount = newMessages.reduce(
        (acc, m) => acc + estimateTokens(m.content),
        0
      );
      return { messages: newMessages, contextTokenCount: newTokenCount };
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
      };
      const newMessages = [...state.messages, message];
      const newTokenCount = newMessages.reduce(
        (acc, m) => acc + estimateTokens(m.content),
        0
      );
      return {
        messages: newMessages,
        contextTokenCount: newTokenCount,
        activeJobIds: state.activeJobIds.filter((id) => id !== job.id),
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

  // Attachment Actions
  addAttachment: (attachment) =>
    set((state) => ({
      uploadedAttachments: [...state.uploadedAttachments, attachment],
    })),

  removeAttachment: (attachmentId) =>
    set((state) => ({
      uploadedAttachments: state.uploadedAttachments.filter(
        (a) => a.id !== attachmentId
      ),
    })),

  clearAttachments: () =>
    set({ uploadedAttachments: [] }),

  // Job Tracking
  trackJob: (jobId) =>
    set((state) => ({
      activeJobIds: [...state.activeJobIds, jobId],
    })),

  untrackJob: (jobId) =>
    set((state) => ({
      activeJobIds: state.activeJobIds.filter((id) => id !== jobId),
    })),

  // Context Management
  getContextForLLM: () => {
    const state = get();
    const maxTokens = state.maxContextTokens;
    const contextItems: ContextItem[] = [];
    let tokenCount = 0;

    // Process messages from newest to oldest, keeping within token limit
    const reversedMessages = [...state.messages].reverse();

    for (const message of reversedMessages) {
      const tokens = estimateTokens(message.content);

      // Skip if adding this message would exceed limit
      if (tokenCount + tokens > maxTokens) {
        break;
      }

      // Add to context (will reverse later)
      contextItems.unshift({
        role: message.role,
        content: message.content,
        tokenEstimate: tokens,
      });

      tokenCount += tokens;
    }

    return contextItems;
  },

  setMaxContextTokens: (tokens) =>
    set({ maxContextTokens: tokens }),
}));
