import { useMutation, useQuery } from '@tanstack/react-query';
import type { UploadMetadata } from '../../types';

export interface UploadResponse {
  id: string;
  filename: string;
  path: string;
  size: [number, number];
  crs: string | null;
  bounds: [number, number, number, number];
  preview_url: string;
  metadata?: UploadMetadata;
}

export interface GeoTIFFFile {
  id: string;
  filename: string;
  path: string;
  size: [number, number];
  crs: string | null;
  bounds: [number, number, number, number];
  size_bytes: number;
  source: 'upload' | 'export';
  modified: number;
  metadata?: UploadMetadata | null;
  upload_id?: string | null;  // what POST /v1/upload or /v1/sample returned; DELETE /v1/files/{upload_id}; null for exports
  kind?: 'upload' | 'sample' | 'export';
  group?: { name: string; line: string } | null;  // both sample files: one Library row
  preview_url?: string | null;
  created_at?: string;  // ISO, server local time
}

/** react-query key of GET /v1/files; Library, map mode and From library share it. */
export const FILES_QUERY_KEY = ['files'] as const;

interface ListFilesResponse {
  ok: boolean;
  files: GeoTIFFFile[];
  count: number;
}

export function useUploadFile() {
  return useMutation({
    mutationFn: async (file: File): Promise<UploadResponse> => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${import.meta.env.VITE_API_URL ?? ''}/v1/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error?.message || error.detail || 'Upload failed');
      }

      return response.json();
    },
  });
}

/** `fresh`: refetch whenever the caller mounts (the Library; files arrive from chat and the sample too). */
export function useListFiles({ enabled = true, fresh = false }: { enabled?: boolean; fresh?: boolean } = {}) {
  return useQuery({
    queryKey: FILES_QUERY_KEY,
    enabled,
    queryFn: async (): Promise<GeoTIFFFile[]> => {
      const response = await fetch(`${import.meta.env.VITE_API_URL ?? ''}/v1/files`);

      if (!response.ok) {
        throw new Error('Failed to fetch files');
      }

      const data: ListFilesResponse = await response.json();
      return data.files;
    },
    staleTime: fresh ? 0 : 30000, // Consider data fresh for 30 seconds
    refetchOnWindowFocus: true,
  });
}
