import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import apiClient from '../client';

export interface DownloadProgress {
  percent: number;
  downloaded_mb: number;
  total_mb: number;
  speed_mbps: number;
  eta_seconds: number | null;
}

export interface MLModel {
  id: string;
  name: string;
  description: string;
  size_mb: number;
  license: string;
  license_url: string;
  huggingface_repo?: string;
  status: 'not_installed' | 'downloading' | 'available' | 'failed';
  path: string | null;
  download_progress?: DownloadProgress;
  error?: string;
}

export interface MLModelsResponse {
  models: MLModel[];
  cache_dir: string;
  total_installed_mb: number;
}

export function useMLModels() {
  const query = useQuery({
    queryKey: ['ml-models'],
    queryFn: () => apiClient.get<MLModelsResponse>('/v1/ml-models'),
    staleTime: 2000, // 2 seconds - refresh often for download progress
  });

  // Check if any model is downloading
  const hasDownloading = query.data?.models?.some((m) => m.status === 'downloading');

  // Auto-refetch when any model is downloading
  useEffect(() => {
    if (!hasDownloading) return;

    const interval = setInterval(() => {
      query.refetch();
    }, 1500); // Poll every 1.5 seconds during download

    return () => clearInterval(interval);
  }, [hasDownloading, query]);

  return query;
}

export function useMLModel(modelId: string) {
  const queryClient = useQueryClient();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const query = useQuery({
    queryKey: ['ml-models', modelId],
    queryFn: () => apiClient.get<MLModel>(`/v1/ml-models/${modelId}`),
    staleTime: 2000,
  });

  // Poll during download
  useEffect(() => {
    if (query.data?.status === 'downloading') {
      intervalRef.current = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['ml-models', modelId] });
        queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      }, 1500);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [query.data?.status, modelId, queryClient]);

  return query;
}

export function useDownloadModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelId: string) =>
      apiClient.post<void, { model_id: string; status: string; message: string }>(
        `/v1/ml-models/${modelId}/download`
      ),
    onSuccess: (_, modelId) => {
      // Invalidate queries to trigger refresh
      queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      queryClient.invalidateQueries({ queryKey: ['ml-models', modelId] });
    },
  });
}

export function useCancelDownload() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelId: string) =>
      apiClient.post<void, { model_id: string; status: string }>(`/v1/ml-models/${modelId}/cancel`),
    onSuccess: (_, modelId) => {
      queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      queryClient.invalidateQueries({ queryKey: ['ml-models', modelId] });
    },
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelId: string) =>
      apiClient.delete<{ model_id: string; status: string; freed_mb: number }>(
        `/v1/ml-models/${modelId}`
      ),
    onSuccess: (_, modelId) => {
      queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      queryClient.invalidateQueries({ queryKey: ['ml-models', modelId] });
    },
  });
}
