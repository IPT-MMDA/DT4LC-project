import { useState, useCallback } from 'react';
import { CloudUpload, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { useUploadFile } from '../../api/hooks/useUpload';

type PeriodType = 'pre' | 'post';

interface UploadState {
  file: File | null;
  uploading: boolean;
  uploaded: boolean;
  error: string | null;
  layerId: string | null;
}

export function CustomDatasetUpload() {
  const { addLayer } = useAppStore();
  const uploadFile = useUploadFile();

  const [preState, setPreState] = useState<UploadState>({
    file: null,
    uploading: false,
    uploaded: false,
    error: null,
    layerId: null,
  });

  const [postState, setPostState] = useState<UploadState>({
    file: null,
    uploading: false,
    uploaded: false,
    error: null,
    layerId: null,
  });

  const handleFileSelect = useCallback(
    (period: PeriodType) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const setState = period === 'pre' ? setPreState : setPostState;
      setState({
        file,
        uploading: false,
        uploaded: false,
        error: null,
        layerId: null,
      });
    },
    []
  );

  const handleUpload = useCallback(
    async (period: PeriodType) => {
      const state = period === 'pre' ? preState : postState;
      const setState = period === 'pre' ? setPreState : setPostState;

      if (!state.file) return;

      // Validate file size (200 MB limit, must match server MAX_UPLOAD_SIZE)
      const MAX_FILE_SIZE = 200 * 1024 * 1024;
      if (state.file.size > MAX_FILE_SIZE) {
        setState((prev) => ({ ...prev, error: 'File is too large. Maximum size is 200 MB.' }));
        return;
      }

      setState((prev) => ({ ...prev, uploading: true, error: null }));

      try {
        const result = await uploadFile.mutateAsync(state.file);

        if (result.path) {
          const layerId = `custom-${period}-${Date.now()}`;
          const layerName = `Custom - ${state.file.name} - ${period === 'pre' ? 'Pre' : 'Post'}`;
          const tileUrl = `/v1/tiles/{z}/{x}/{y}?path=${encodeURIComponent(result.path)}`;

          addLayer({
            id: layerId,
            name: layerName,
            type: 'raster',
            visible: true,
            opacity: 1.0,
            url: tileUrl,
          });

          setState((prev) => ({ ...prev, uploading: false, uploaded: true, layerId }));
        } else {
          throw new Error('Upload failed');
        }
      } catch (error) {
        setState((prev) => ({
          ...prev,
          uploading: false,
          error: error instanceof Error ? error.message : 'Failed to upload file',
        }));
      }
    },
    [preState, postState, uploadFile, addLayer]
  );

  const renderUploadZone = (period: PeriodType) => {
    const state = period === 'pre' ? preState : postState;
    const label = period === 'pre' ? 'Pre-Event Data' : 'Post-Event Data';

    return (
      <div className="mb-4">
        <div className="text-sm font-medium text-text2 mb-2">{label}</div>
        <div
          className={`p-4 text-center border-2 border-dashed rounded-lg ${
            state.uploaded
              ? 'border-done bg-done-tint'
              : state.error
              ? 'border-fail bg-fail-tint'
              : 'border-border bg-surface2'
          }`}
        >
          {state.uploaded ? (
            <div>
              <CheckCircle2 className="w-10 h-10 text-done mx-auto mb-2" />
              <div className="text-sm text-done-ink">Uploaded: {state.file?.name}</div>
              <div className="text-xs text-done">Layer added to map</div>
            </div>
          ) : (
            <div>
              <CloudUpload className="w-10 h-10 text-faint mx-auto mb-2" />
              <div className="text-sm text-muted mb-2">
                {state.file ? state.file.name : 'No file selected'}
              </div>

              <label className="inline-block px-3 py-1 text-sm border border-border rounded cursor-pointer hover:bg-surface2">
                {state.file ? 'Change File' : 'Select GeoTIFF'}
                <input type="file" accept=".tif,.tiff" className="hidden" onChange={handleFileSelect(period)} />
              </label>

              {state.file && !state.uploaded && (
                <button
                  onClick={() => handleUpload(period)}
                  disabled={state.uploading}
                  className="ml-2 px-3 py-1 text-sm bg-accent text-on-accent rounded hover:bg-accent/90 disabled:bg-faint disabled:cursor-not-allowed"
                >
                  {state.uploading ? (
                    <span className="flex items-center gap-1">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      Uploading...
                    </span>
                  ) : (
                    'Upload'
                  )}
                </button>
              )}

              {state.uploading && (
                <div className="mt-2 w-full bg-border rounded-full h-1">
                  <div className="bg-accent h-1 rounded-full animate-pulse" style={{ width: '50%' }} />
                </div>
              )}
            </div>
          )}

          {state.error && (
            <div className="flex items-start gap-2 mt-2 p-2 bg-fail-tint border border-fail/30 rounded">
              <AlertCircle className="w-4 h-4 text-fail mt-0.5 flex-shrink-0" />
              <p className="text-sm text-fail-ink">{state.error}</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div>
      <p className="text-sm text-muted mb-4">
        Upload your own GeoTIFF files for pre and post periods
      </p>

      {renderUploadZone('pre')}
      {renderUploadZone('post')}

      <div className="mt-4 p-3 bg-accent-tint border border-accent/30 rounded text-xs text-accent-ink">
        Supported format: GeoTIFF (.tif, .tiff) with valid CRS and georeferencing
      </div>
    </div>
  );
}
