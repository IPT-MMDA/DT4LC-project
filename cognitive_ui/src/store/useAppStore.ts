import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/** WGS84 [west, south, east, north]. */
export type LngLatBounds = [number, number, number, number];

/** The colour key a result layer shows in the docked map's Layers panel. */
export type LayerLegend =
  | { kind: 'change' }  // lost · same · gained, the legend-* tokens
  | { kind: 'ramp'; stops: string[]; range: [number, number] };  // an index colour ramp

export interface Layer {
  id: string;
  name: string;
  type: string;
  visible: boolean;
  opacity: number;
  url?: string; // Optional URL for GeoTIFF layers
  bounds?: LngLatBounds;  // result layers: what Fit to result zooms to
  legend?: LayerLegend;
}

export interface DrawnRegion {
  id: string;
  name: string;
  type: 'bbox' | 'polygon';
  geometry: GeoJSON.Feature;
  bbox: [number, number, number, number]; // [minX, minY, maxX, maxY]
  createdAt: string;
}

interface AppStore {
  // UI State
  theme: 'light' | 'dark' | 'auto';
  useNewDatasetPanel: boolean; // Feature flag for new dataset selection panel

  // Map State
  currentBasemap: string;
  mapLayers: Layer[];
  selectedFeature: any | null;
  mapInstance: any | null; // MapLibre map instance
  fitRequest: LngLatBounds | null;  // bounds the mounted map should zoom to next

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

  // Actions
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
  /** Add a result layer, or replace the one with the same id, and zoom the map to it. */
  addResultLayer: (layer: Layer) => void;
  requestFit: (bounds: LngLatBounds) => void;
  clearFitRequest: () => void;
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
    (set) => {
      const defaultDates = getDefaultDates();

      return {
      // Initial state
      theme: 'light',
      useNewDatasetPanel: true, // Default to new panel for new users
      currentBasemap: 'satellite',
      mapLayers: [],
      selectedFeature: null,
      mapInstance: null,
      fitRequest: null,
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

  // UI Actions
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

  addResultLayer: (layer) =>
    set((state) => ({
      mapLayers: [...state.mapLayers.filter((l) => l.id !== layer.id), layer],
      fitRequest: layer.bounds ?? state.fitRequest,
    })),

  requestFit: (bounds) =>
    set({ fitRequest: bounds }),

  clearFitRequest: () =>
    set({ fitRequest: null }),

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

    };
  },
    {
      name: 'dt4lc-app-storage',
      partialize: (state) => ({
        drawnRegions: state.drawnRegions,
        selectedRegion: state.selectedRegion,
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
