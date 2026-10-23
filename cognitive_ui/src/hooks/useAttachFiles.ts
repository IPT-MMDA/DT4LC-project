import { useCallback, useState } from 'react';
import { useUploadFile, type UploadResponse } from '../api/hooks/useUpload';
import { useChatStore } from '../store/useChatStore';
import type { Attachment } from '../types';

const GEOTIFF = /\.tiff?$/i;

/** An upload (POST /v1/upload) as a composer attachment, metadata kept for chat and jobs. */
export function uploadAttachment(upload: UploadResponse): Attachment {
  return {
    id: upload.id,
    filename: upload.filename,
    path: upload.path,
    mime_type: 'image/tiff',
    size_bytes: upload.metadata?.size_bytes,
    preview_url: upload.preview_url,
    metadata: upload.metadata,
  };
}

/**
 * Upload GeoTIFFs (the composer's +, the empty state's Upload GeoTIFF, the drop
 * zone) and add each to the composer. Other files are refused with a message.
 */
export function useAttachFiles() {
  const { mutateAsync: uploadFile } = useUploadFile();
  const [uploading, setUploading] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const attachFiles = useCallback(
    async (files: Iterable<File>) => {
      const list = [...files];
      const rejected = list.filter((f) => !GEOTIFF.test(f.name));
      setError(rejected.length > 0 ? `Not a GeoTIFF: ${rejected.map((f) => f.name).join(', ')}` : null);
      for (const file of list.filter((f) => GEOTIFF.test(f.name))) {
        setUploading((n) => n + 1);
        try {
          useChatStore.getState().addAttachment(uploadAttachment(await uploadFile(file)));
        } catch (err) {
          setError(`Upload failed: ${file.name}: ${err instanceof Error ? err.message : String(err)}`);
        } finally {
          setUploading((n) => n - 1);
        }
      }
    },
    [uploadFile],
  );

  return { attachFiles, uploading: uploading > 0, error, clearError: () => setError(null) };
}
