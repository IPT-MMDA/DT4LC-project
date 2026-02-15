import { useMutation, useQuery } from '@tanstack/react-query';
import apiClient from '../client';

interface FetchSentinel2Request {
  bbox: [number, number, number, number];
  start_date: string;
  end_date: string;
  data_type: 'rgb' | 'ndvi' | 'ndwi' | 'ndsi';
  cloud_cover_max?: number;
}

interface Sentinel2Response {
  ok: boolean;
  tile_url?: string;
  image_count?: number;
  bbox?: [number, number, number, number];
  start_date?: string;
  end_date?: string;
  cloud_cover_max?: number;
  data_type?: string;
  index_type?: string;
  vis_params?: Record<string, any>;
  error?: string;
}

interface AvailableDatesRequest {
  bbox: [number, number, number, number];
  start_date: string;
  end_date: string;
  cloud_cover_max?: number;
}

interface AvailableDatesResponse {
  ok: boolean;
  dates?: string[];
  count?: number;
  bbox?: [number, number, number, number];
  start_date?: string;
  end_date?: string;
  error?: string;
}

export function useFetchSentinel2() {
  return useMutation<Sentinel2Response, Error, FetchSentinel2Request>({
    mutationFn: async (data) => {
      const params = new URLSearchParams({
        start_date: data.start_date,
        end_date: data.end_date,
        data_type: data.data_type,
        cloud_cover_max: (data.cloud_cover_max ?? 20).toString(),
      });

      return apiClient.post<[number, number, number, number], Sentinel2Response>(
        `/v1/gee/sentinel2?${params}`,
        data.bbox
      );
    },
  });
}

export function useAvailableDates(params: AvailableDatesRequest | null) {
  return useQuery<AvailableDatesResponse, Error>({
    queryKey: ['gee', 'dates', params],
    queryFn: async () => {
      if (!params) throw new Error('No parameters provided');

      const queryParams = new URLSearchParams({
        bbox: JSON.stringify(params.bbox),
        start_date: params.start_date,
        end_date: params.end_date,
        cloud_cover_max: (params.cloud_cover_max ?? 20).toString(),
      });

      return apiClient.get<AvailableDatesResponse>(
        `/v1/gee/dates?${queryParams}`
      );
    },
    enabled: !!params,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// MODIS dataset hooks
export function useFetchMODIS() {
  return useMutation<Sentinel2Response, Error, FetchSentinel2Request>({
    mutationFn: async (data) => {
      const params = new URLSearchParams({
        start_date: data.start_date,
        end_date: data.end_date,
        data_type: data.data_type,
        cloud_cover_max: (data.cloud_cover_max ?? 20).toString(),
      });

      return apiClient.post<[number, number, number, number], Sentinel2Response>(
        `/v1/gee/modis?${params}`,
        data.bbox
      );
    },
  });
}

// Landsat dataset hooks
export function useFetchLandsat() {
  return useMutation<Sentinel2Response, Error, FetchSentinel2Request>({
    mutationFn: async (data) => {
      const params = new URLSearchParams({
        start_date: data.start_date,
        end_date: data.end_date,
        data_type: data.data_type,
        cloud_cover_max: (data.cloud_cover_max ?? 20).toString(),
      });

      return apiClient.post<[number, number, number, number], Sentinel2Response>(
        `/v1/gee/landsat?${params}`,
        data.bbox
      );
    },
  });
}

// Bulk fetch types and hooks
export interface BulkFetchRequest {
  bbox: [number, number, number, number];
  dataset_id: string;
  bands: string[];
  indices: string[];
  pre_start: string;
  pre_end: string;
  post_start: string | null;
  post_end: string | null;
  cloud_cover_max: number;
  use_now: boolean;
}

export interface LayerMetadata {
  tile_url: string;
  layer_name: string;
  layer_id: string;
  period: 'pre' | 'post';
  data_type: 'bands' | 'ndvi' | 'ndwi' | 'ndsi';
  bands?: string[];
  index_type?: string;
  image_count?: number;
  vis_params?: Record<string, any>;
}

export interface BulkFetchResponse {
  ok: boolean;
  layers: LayerMetadata[];
  total_layers?: number;
  errors?: string[] | null;
  error?: string;
}

export function useBulkFetchDatasets() {
  return useMutation<BulkFetchResponse, Error, BulkFetchRequest>({
    mutationFn: async (data) => {
      return apiClient.post<BulkFetchRequest, BulkFetchResponse>(
        '/v1/gee/bulk-fetch',
        data
      );
    },
  });
}

// Layer metadata persistence
export interface PersistLayerRequest {
  layer_id: string;
  layer_name: string;
  dataset_id: string;
  bands: string[];
  indices: string[];
  period: string;
  bbox: [number, number, number, number];
  start_date: string;
  end_date: string;
  tile_url: string;
  cloud_cover_max: number;
}

export function usePersistLayer() {
  return useMutation<{ ok: boolean; layer_id: string }, Error, PersistLayerRequest>({
    mutationFn: async (data) => {
      return apiClient.post('/v1/gee/layers/persist', data);
    },
  });
}

// Export layer for analysis
export interface ExportLayerRequest {
  layer_id: string;
  scale?: number;
  format?: string;
  source?: 'auto' | 'gee' | 'microsoft';
  name?: string;
}

export interface ExportLayerResponse {
  ok: boolean;
  status: string;
  attachment: {
    id: string;
    filename: string;
    path: string;
    mime_type: string;
    size_bytes: number;
    source: string;
    metadata: Record<string, any>;
  };
  size_mb: number;
  source?: string;
  error?: string;
}

export function useExportLayer() {
  return useMutation<ExportLayerResponse, Error, ExportLayerRequest>({
    mutationFn: async ({ layer_id, scale = 10, format = 'geotiff', source = 'auto', name }) => {
      const params = new URLSearchParams({
        scale: scale.toString(),
        format: format,
        source: source,
      });

      if (name) {
        params.set('name', name);
      }

      return apiClient.post(`/v1/gee/layers/${layer_id}/export?${params}`, {});
    },
  });
}
