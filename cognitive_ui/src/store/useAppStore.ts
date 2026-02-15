import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage, Attachment, ContextItem, JobResultData, Job } from '../types';

interface Layer {
  id: string;
  name: string;
  type: string;
  visible: boolean;
  opacity: number;
  url?: string; // Optional URL for GeoTIFF layers
}

export interface DrawnRegion {
  id: string;
  name: string;
  type: 'bbox' | 'polygon';
  geometry: GeoJSON.Feature;
  bbox: [number, number, number, number]; // [minX, minY, maxX, maxY]
  createdAt: string;
}

// Context management settings
const MAX_CONTEXT_TOKENS = 4000; // Safe limit for small LLM instances
const CHARS_PER_TOKEN = 4; // Rough estimate

interface AppStore {
  // UI State
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark' | 'auto';
  useNewDatasetPanel: boolean; // Feature flag for new dataset selection panel

  // Map State
  currentBasemap: string;
  mapLayers: Layer[];
  selectedFeature: any | null;
  mapInstance: any | null; // MapLibre map instance

  // Map Regions
  drawnRegions: DrawnRegion[];
  selectedRegion: DrawnRegion | null;

  // Dataset Selection State
  selectedDatasetId: string; // 'sentinel-2' | 'modis' | 'landsat-8' | 'custom'
  selectedBands: string[];
  selectedIndices: ('ndvi' | 'ndwi' | 'ndsi')[];
  prePeriod: { start: string; end: string };
  postPeriod: { start: string; end: string };
  useNowForPost: boolean;
  cloudCoverMax: number;
  isFetchingDatasets: boolean;
  fetchError: string | null;

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
  toggleDatasetPanelVersion: () => void;

  // Dataset Selection Actions
  setSelectedDataset: (datasetId: string) => void;
  toggleBand: (bandId: string) => void;
  setBands: (bands: string[]) => void;
  toggleIndex: (index: 'ndvi' | 'ndwi' | 'ndsi') => void;
  setPrePeriod: (start: string, end: string) => void;
  setPostPeriod: (start: string, end: string) => void;
  setUseNowForPost: (useNow: boolean) => void;
  setCloudCoverMax: (max: number) => void;
  setIsFetchingDatasets: (isFetching: boolean) => void;
  setFetchError: (error: string | null) => void;
  resetSelection: () => void;

  // Map Actions
  setBasemap: (basemap: string) => void;
  setMapInstance: (instance: any | null) => void;
  addLayer: (layer: Layer) => void;
  removeLayer: (layerId: string) => void;
  toggleLayerVisibility: (layerId: string) => void;
  setLayerOpacity: (layerId: string, opacity: number) => void;
  setSelectedFeature: (feature: any | null) => void;

  // Map Region Actions
  addRegion: (region: DrawnRegion) => void;
  updateRegion: (id: string, updates: Partial<DrawnRegion>) => void;
  deleteRegion: (id: string) => void;
  setSelectedRegion: (region: DrawnRegion | null) => void;
  clearRegions: () => void;

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

  if (result.reconstruction) {
    parts.push(`[Prithvi reconstruction: ${result.reconstruction.model}]`);
  }

  return parts.join(' ');
}

// Helper to calculate default dates
function getDefaultDates() {
  const today = new Date();
  const thirtyDaysAgo = new Date(today);
  thirtyDaysAgo.setDate(today.getDate() - 30);
  const sevenDaysAgo = new Date(today);
  sevenDaysAgo.setDate(today.getDate() - 7);

  return {
    preStart: thirtyDaysAgo.toISOString().split('T')[0],
    preEnd: today.toISOString().split('T')[0],
    postStart: sevenDaysAgo.toISOString().split('T')[0],
    postEnd: today.toISOString().split('T')[0],
  };
}

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => {
      const defaultDates = getDefaultDates();

      return {
      // Initial state
      sidebarCollapsed: false,
      theme: 'light',
      useNewDatasetPanel: true, // Default to new panel for new users
      currentBasemap: 'satellite',
      mapLayers: [],
      selectedFeature: null,
      mapInstance: null,
      drawnRegions: [],
      selectedRegion: null,

      // Dataset Selection Initial State
      selectedDatasetId: 'sentinel-2',
      selectedBands: ['B2', 'B3', 'B4', 'B8', 'B11'], // Include NIR+SWIR for indices
      selectedIndices: [],
      prePeriod: { start: defaultDates.preStart, end: defaultDates.preEnd },
      postPeriod: { start: defaultDates.postStart, end: defaultDates.postEnd },
      useNowForPost: true,
      cloudCoverMax: 20,
      isFetchingDatasets: false,
      fetchError: null,

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

  toggleDatasetPanelVersion: () =>
    set((state) => ({ useNewDatasetPanel: !state.useNewDatasetPanel })),

  // Dataset Selection Actions
  setSelectedDataset: (datasetId) =>
    set({ selectedDatasetId: datasetId }),

  toggleBand: (bandId) =>
    set((state) => {
      const bands = state.selectedBands.includes(bandId)
        ? state.selectedBands.filter((b) => b !== bandId)
        : [...state.selectedBands, bandId];
      return { selectedBands: bands };
    }),

  setBands: (bands) =>
    set({ selectedBands: bands }),

  toggleIndex: (index) =>
    set((state) => {
      const indices = state.selectedIndices.includes(index)
        ? state.selectedIndices.filter((i) => i !== index)
        : [...state.selectedIndices, index];
      return { selectedIndices: indices };
    }),

  setPrePeriod: (start, end) =>
    set({ prePeriod: { start, end } }),

  setPostPeriod: (start, end) =>
    set({ postPeriod: { start, end } }),

  setUseNowForPost: (useNow) =>
    set({ useNowForPost: useNow }),

  setCloudCoverMax: (max) =>
    set({ cloudCoverMax: max }),

  setIsFetchingDatasets: (isFetching) =>
    set({ isFetchingDatasets: isFetching }),

  setFetchError: (error) =>
    set({ fetchError: error }),

  resetSelection: () => {
    const defaultDates = getDefaultDates();
    set({
      selectedDatasetId: 'sentinel-2',
      selectedBands: ['B2', 'B3', 'B4', 'B8', 'B11'], // Include NIR+SWIR for indices
      selectedIndices: [],
      prePeriod: { start: defaultDates.preStart, end: defaultDates.preEnd },
      postPeriod: { start: defaultDates.postStart, end: defaultDates.postEnd },
      useNowForPost: true,
      cloudCoverMax: 20,
      isFetchingDatasets: false,
      fetchError: null,
    });
  },

  // Map Actions
  setBasemap: (basemap) =>
    set({ currentBasemap: basemap }),

  setMapInstance: (instance) =>
    set({ mapInstance: instance }),

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

  // Map Region Actions
  addRegion: (region) =>
    set((state) => ({
      drawnRegions: [...state.drawnRegions, region],
      selectedRegion: region
    })),

  updateRegion: (id, updates) =>
    set((state) => ({
      drawnRegions: state.drawnRegions.map((r) =>
        r.id === id ? { ...r, ...updates } : r
      ),
    })),

  deleteRegion: (id) =>
    set((state) => ({
      drawnRegions: state.drawnRegions.filter((r) => r.id !== id),
      selectedRegion: state.selectedRegion?.id === id ? null : state.selectedRegion
    })),

  setSelectedRegion: (region) =>
    set({ selectedRegion: region }),

  clearRegions: () =>
    set({ drawnRegions: [], selectedRegion: null }),

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
    };
  },
    {
      name: 'dt4lc-app-storage',
      partialize: (state) => ({
        drawnRegions: state.drawnRegions,
        currentBasemap: state.currentBasemap,
        theme: state.theme,
        useNewDatasetPanel: state.useNewDatasetPanel,
        selectedDatasetId: state.selectedDatasetId,
        prePeriod: state.prePeriod,
        postPeriod: state.postPeriod,
        useNowForPost: state.useNowForPost,
        cloudCoverMax: state.cloudCoverMax,
      }),
    }
  )
);
