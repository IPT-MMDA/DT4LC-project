import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
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
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (file: File): Promise<UploadResponse> => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${import.meta.env.VITE_API_URL ?? ''}/v1/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // nginx answers an oversized body (413) or a restarting API (502) with an HTML page.
        const error = await response.json().catch(() => null);
        const message = error?.error?.message || error?.detail;
        if (message) throw new Error(message);
        throw new Error(response.status === 413 ? 'File too large for this server (HTTP 413)' : `Upload failed (HTTP ${response.status})`);
      }

      return response.json();
    },
    // The upload is among the caller's files now (From library, map mode, the Library).
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: FILES_QUERY_KEY });
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
