import { useMutation } from '@tanstack/react-query';
import apiClient from '../client';
import type { UploadMetadata } from '../../types';

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
  vis_params?: Record<string, unknown>;
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
    metadata: UploadMetadata;
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
