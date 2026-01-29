import { useMutation, useQuery } from '@tanstack/react-query';

interface UploadResponse {
  id: string;
  filename: string;
  path: string;
  size: [number, number];
  crs: string | null;
  bounds: [number, number, number, number];
  preview_png_base64: string;
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
}

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

      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/v1/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      return response.json();
    },
  });
}

export function useListFiles() {
  return useQuery({
    queryKey: ['files'],
    queryFn: async (): Promise<GeoTIFFFile[]> => {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/v1/files`);

      if (!response.ok) {
        throw new Error('Failed to fetch files');
      }

      const data: ListFilesResponse = await response.json();
      return data.files;
    },
    staleTime: 30000, // Consider data fresh for 30 seconds
    refetchOnWindowFocus: true,
  });
}
