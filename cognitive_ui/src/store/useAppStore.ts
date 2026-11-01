import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { RegionDataset } from '../types';

/** WGS84 [west, south, east, north]. */
export type LngLatBounds = [number, number, number, number];

/** A class of a change result with its area: "lost 0.6 km²" in the map legend. */
export interface LegendClass {
  label: string;  // lost, no change, gained; wetter, drier; increase, stable
  tone: 'gained' | 'lost' | 'neutral';
  colour?: string;  // the class colour from `class_statistics`
  area_km2?: number;
}

/** The colour key a result layer shows in the Layers panel and the map legend. */
export type LayerLegend =
  // lost · same · gained: the raster style's ramp (else the legend-* tokens), its range and the class areas
  | { kind: 'change'; stops?: string[]; range?: [number, number]; classes?: LegendClass[] }
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
  source?: 'chat' | 'sample' | 'earth-engine';  // where it came from: a result card's On map, the sample, Fetch to map
  subtitle?: string;  // map mode's second line, e.g. "HLS L30 · true colour · 2023-06-09"
  addedAt?: string;  // ISO 8601: "from chat · just now"
  footprint?: [number, number][];  // the raster's WGS84 corners (metadata.footprint_wgs84), drawn as its outline
  thumbnail?: string;  // server-relative preview URL: the swatch of imagery without a ramp
}

/** Map mode's drawing tool: Pan, Polygon or Rectangle. */
export type MapTool = 'pan' | 'polygon' | 'rectangle';

/** The imagery panel's request: the region fields of POST /v1/jobs and /v1/chat. */
export interface Imagery {
  dataset: RegionDataset;
  start_date: string;  // YYYY-MM-DD
  end_date: string;
  cloud_max: number;  // percent
  bands: string[];  // band codes of the dataset
}

export interface MapView {
  longitude: number;
  latitude: number;
  zoom: number;
}

/** Compare: `left` shows left of the divider, `right` right of it; `split` is the divider's place, 0..1. */
export interface CompareState {
  left: string;
  right: string;
  split: number;
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

  // Map State
  currentBasemap: string;
  mapLayers: Layer[];
  selectedFeature: GeoJSON.Feature | null;
  fitRequest: LngLatBounds | null;  // bounds the mounted map should zoom to next
  mapTool: MapTool;
  regionOpacity: number;  // the drawn regions' layer row, 0..1
  compare: CompareState | null;  // null: no divider
  mapView: MapView | null;  // the last view of either map, so map mode opens where the docked map was

  // Map Regions
  drawnRegions: DrawnRegion[];
  selectedRegion: DrawnRegion | null;

  // Map mode's imagery panel; null until edited (the panel shows its defaults)
  imagery: Imagery | null;

  // Actions
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  setImagery: (imagery: Imagery) => void;

  // Map Actions
  setBasemap: (basemap: string) => void;
  addLayer: (layer: Layer) => void;
  /** Add a layer under all others unless one with its id is there (the sample's true colour). */
  addBaseLayer: (layer: Layer) => void;
  /** Add a result layer, or replace the one with the same id, and zoom the map to it. */
  addResultLayer: (layer: Layer) => void;
  requestFit: (bounds: LngLatBounds) => void;
  clearFitRequest: () => void;
  removeLayer: (layerId: string) => void;
  toggleLayerVisibility: (layerId: string) => void;
  setLayerOpacity: (layerId: string, opacity: number) => void;
  setRegionOpacity: (opacity: number) => void;
  setSelectedFeature: (feature: GeoJSON.Feature | null) => void;
  setMapTool: (tool: MapTool) => void;
  setCompare: (compare: CompareState | null) => void;
  setMapView: (view: MapView) => void;

  // Map Region Actions
  addRegion: (region: DrawnRegion) => void;
  /** Also updates the selected region when it is this one (Rename). */
  updateRegion: (id: string, updates: Partial<DrawnRegion>) => void;
  deleteRegion: (id: string) => void;
  setSelectedRegion: (region: DrawnRegion | null) => void;
  clearRegions: () => void;

}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      // Initial state
      theme: 'light',
      currentBasemap: 'satellite',
      mapLayers: [],
      selectedFeature: null,
      fitRequest: null,
      mapTool: 'pan',
      regionOpacity: 1,
      compare: null,
      mapView: null,
      drawnRegions: [],
      selectedRegion: null,
      imagery: null,

  // UI Actions
  setTheme: (theme) =>
    set({ theme }),

  setImagery: (imagery) =>
    set({ imagery }),

  // Map Actions
  setBasemap: (basemap) =>
    set({ currentBasemap: basemap }),

  addLayer: (layer) =>
    set((state) => ({ mapLayers: [...state.mapLayers, layer] })),

  addBaseLayer: (layer) =>
    set((state) => (state.mapLayers.some((l) => l.id === layer.id) ? {} : { mapLayers: [layer, ...state.mapLayers] })),

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

  setRegionOpacity: (opacity) =>
    set({ regionOpacity: opacity }),

  setSelectedFeature: (feature) =>
    set({ selectedFeature: feature }),

  setMapTool: (tool) =>
    set({ mapTool: tool }),

  setCompare: (compare) =>
    set({ compare }),

  setMapView: (view) =>
    set({ mapView: view }),

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
      selectedRegion: state.selectedRegion?.id === id ? { ...state.selectedRegion, ...updates } : state.selectedRegion,
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

    }),
    {
      name: 'dt4lc-app-storage',
      partialize: (state) => ({
        drawnRegions: state.drawnRegions,
        selectedRegion: state.selectedRegion,
        currentBasemap: state.currentBasemap,
        theme: state.theme,
        imagery: state.imagery,
      }),
    }
  )
);
