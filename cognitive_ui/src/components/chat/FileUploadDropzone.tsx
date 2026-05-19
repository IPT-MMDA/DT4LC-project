import { useState, useRef, useCallback } from 'react';
import { Upload, X, Loader2, AlertCircle, CheckCircle, Paperclip, Download, Database } from 'lucide-react';
import type { GeoTIFFFile } from '../../api/hooks/useUpload';
import { useUploadFile, useListFiles } from '../../api/hooks/useUpload';
import { useAppStore } from '../../store/useAppStore';
import { uploadLogger as logger } from '../../utils/logger';

interface FileUploadDropzoneProps {
  onClose?: () => void;
  compact?: boolean;
}

interface UploadedFilePreview {
  id: string;
  filename: string;
  preview_png_base64?: string;
  size: [number, number];
  crs: string | null;
}

export function FileUploadDropzone({ onClose, compact = false }: FileUploadDropzoneProps) {
  const [dragActive, setDragActive] = useState(false);
  const [recentUpload, setRecentUpload] = useState<UploadedFilePreview | null>(null);
  const [showAvailableFiles, setShowAvailableFiles] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadFile = useUploadFile();
  const { data: availableFiles, isLoading: isLoadingFiles, isError: isFilesError } = useListFiles();
  const { addAttachment, uploadedAttachments } = useAppStore();

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  const isAttached = (file: GeoTIFFFile) => {
    return uploadedAttachments.some(att => att.id === file.id);
  };

  const attachAvailableFile = (file: GeoTIFFFile) => {
    if (isAttached(file)) return;

    addAttachment({
      id: file.id,
      filename: file.filename,
      path: file.path,
      mime_type: 'image/tiff',
      size_bytes: file.size_bytes,
    });

    if (onClose) onClose();
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const handleFiles = async (files: FileList) => {
    const file = files[0];

    // Validate file type
    const validExtensions = ['.tif', '.tiff', '.geotiff'];
    const isValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!isValid) {
      alert('Please upload a GeoTIFF file (.tif, .tiff)');
      return;
    }

    // Validate file size (200 MB limit, must match server MAX_UPLOAD_SIZE)
    const MAX_FILE_SIZE = 200 * 1024 * 1024;
    if (file.size > MAX_FILE_SIZE) {
      alert('File is too large. Maximum size is 200 MB.');
      return;
    }

    try {
      const result = await uploadFile.mutateAsync(file);

      logger.debug('Upload result:', {
        id: result.id,
        filename: result.filename,
        path: result.path,
        hasPath: !!result.path,
      });

      // Store preview for display
      setRecentUpload({
        id: result.id,
        filename: result.filename,
        preview_png_base64: result.preview_png_base64,
        size: result.size,
        crs: result.crs,
      });

      // Add to global attachments store (including preview for chat display)
      const attachment = {
        id: result.id,
        filename: result.filename,
        path: result.path,
        mime_type: 'image/tiff',
        preview_png_base64: result.preview_png_base64,
      };
      logger.debug('Adding attachment to store:', attachment);
      addAttachment(attachment);

      // Auto-close after successful upload if compact mode
      if (compact && onClose) {
        setTimeout(() => onClose(), 1500);
      }
    } catch (error) {
      logger.error('Upload error:', error);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  const renderAvailableFiles = () => (
    <div className="space-y-2">
      {isLoadingFiles ? (
        <div className="flex items-center justify-center py-6">
          <Loader2 className="w-5 h-5 text-primary-500 animate-spin" />
        </div>
      ) : isFilesError ? (
        <div className="flex items-center gap-2 py-3 text-sm text-red-600 dark:text-red-400">
          <AlertCircle className="w-4 h-4" />
          Failed to load files
        </div>
      ) : availableFiles && availableFiles.length > 0 ? (
        <>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
            Select a file to add to chat:
          </p>
          {availableFiles.map((file) => (
            <button
              key={file.id}
              type="button"
              onClick={() => attachAvailableFile(file)}
              disabled={isAttached(file)}
              className="w-full text-left px-3 py-2 bg-gray-50 dark:bg-gray-950 hover:bg-gray-100 dark:hover:bg-gray-900 disabled:opacity-60 rounded border border-gray-200 dark:border-gray-800 transition-colors"
            >
              <div className="flex items-start gap-2">
                {file.source === 'export' ? (
                  <Download className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                ) : (
                  <Upload className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {file.filename}
                    </p>
                    {isAttached(file) && (
                      <CheckCircle className="w-3 h-3 text-green-500 flex-shrink-0" />
                    )}
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {file.source} | {formatFileSize(file.size_bytes)} | {formatDate(file.modified)}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </>
      ) : (
        <div className="text-center py-6 text-gray-400">
          <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-xs">No files available</p>
        </div>
      )}
    </div>
  );

  // Compact inline version
  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex gap-2">
          <div
            className={`flex-1 rounded-lg border-2 border-dashed p-4 transition-colors ${
              dragActive
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-950'
                : uploadFile.isError
                ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-950'
                : 'border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff,.geotiff"
              onChange={handleChange}
              className="hidden"
            />

            {uploadFile.isPending ? (
              <div className="flex items-center gap-3">
                <Loader2 className="w-5 h-5 text-primary-500 animate-spin" />
                <span className="text-sm text-gray-600 dark:text-gray-400">Uploading...</span>
              </div>
            ) : uploadFile.isError ? (
              <div className="flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <span className="text-sm text-red-600 dark:text-red-400">
                  {uploadFile.error instanceof Error ? uploadFile.error.message : 'Upload failed'}
                </span>
                <button
                  onClick={() => uploadFile.reset()}
                  className="ml-auto text-xs text-primary-500 hover:text-primary-600"
                >
                  Try again
                </button>
              </div>
            ) : recentUpload ? (
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {recentUpload.filename}
                  </p>
                  <p className="text-xs text-gray-500">
                    {recentUpload.size[0]} x {recentUpload.size[1]} px
                  </p>
                </div>
                {recentUpload.preview_png_base64 && (
                  <img
                    src={`data:image/png;base64,${recentUpload.preview_png_base64}`}
                    alt="Preview"
                    className="w-12 h-12 rounded object-cover"
                  />
                )}
              </div>
            ) : (
              <div
                className="flex items-center gap-3 cursor-pointer"
                onClick={onButtonClick}
              >
                <Upload className="w-5 h-5 text-gray-400" />
                <div className="flex-1">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    Drop a GeoTIFF here or <span className="text-primary-500">browse</span>
                  </p>
                </div>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => setShowAvailableFiles(!showAvailableFiles)}
            className={`px-4 rounded-lg transition-colors flex items-center gap-2 ${
              showAvailableFiles
                ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
            title="Choose an existing file"
          >
            <Database className="w-5 h-5" />
          </button>
        </div>

        {showAvailableFiles && (
          <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 shadow-lg p-3 max-h-64 overflow-y-auto">
            {renderAvailableFiles()}
          </div>
        )}
      </div>
    );
  }

  // Full dropdown version with preview
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 shadow-lg p-4 w-80 max-h-[80vh] overflow-y-auto">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-900 dark:text-white">
          {showAvailableFiles ? 'Available Files' : 'Upload GeoTIFF'}
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowAvailableFiles(!showAvailableFiles)}
            className="text-xs text-primary-500 hover:text-primary-600"
          >
            {showAvailableFiles ? 'Upload' : 'Browse'}
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Current attachments */}
      {uploadedAttachments.length > 0 && (
        <div className="mb-3 space-y-2">
          <p className="text-xs text-gray-500 dark:text-gray-400">Attached files:</p>
          {uploadedAttachments.map((att) => (
            <div
              key={att.id}
              className="flex items-center gap-2 px-2 py-1 bg-primary-50 dark:bg-primary-950 rounded text-sm"
            >
              <Paperclip className="w-3 h-3 text-primary-500" />
              <span className="flex-1 truncate text-primary-700 dark:text-primary-300">
                {att.filename}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Available files list */}
      {showAvailableFiles && renderAvailableFiles()}

      {/* Upload zone - only show when not browsing */}
      {!showAvailableFiles && (
        <div
          className={`rounded-lg border-2 border-dashed p-6 transition-colors ${
            dragActive
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-950'
              : uploadFile.isError
              ? 'border-red-300 dark:border-red-700'
              : 'border-gray-300 dark:border-gray-700'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.geotiff"
          onChange={handleChange}
          className="hidden"
        />

        <div className="text-center">
          {uploadFile.isPending ? (
            <>
              <Loader2 className="w-8 h-8 mx-auto text-primary-500 mb-2 animate-spin" />
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Processing GeoTIFF...
              </p>
            </>
          ) : uploadFile.isError ? (
            <>
              <AlertCircle className="w-8 h-8 mx-auto text-red-500 mb-2" />
              <p className="text-sm text-red-600 dark:text-red-400 mb-2">
                {uploadFile.error instanceof Error ? uploadFile.error.message : 'Upload failed'}
              </p>
              <button
                onClick={() => uploadFile.reset()}
                className="text-xs text-primary-500 hover:text-primary-600"
              >
                Try again
              </button>
            </>
          ) : recentUpload ? (
            <>
              <CheckCircle className="w-8 h-8 mx-auto text-green-500 mb-2" />
              <p className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                {recentUpload.filename}
              </p>
              <p className="text-xs text-gray-500">
                {recentUpload.size[0]} x {recentUpload.size[1]} px
                {recentUpload.crs && ` | ${recentUpload.crs}`}
              </p>
              {recentUpload.preview_png_base64 && (
                <img
                  src={`data:image/png;base64,${recentUpload.preview_png_base64}`}
                  alt="Preview"
                  className="mt-2 mx-auto w-full h-24 rounded object-cover"
                />
              )}
              <button
                onClick={() => {
                  setRecentUpload(null);
                  uploadFile.reset();
                }}
                className="mt-2 text-xs text-primary-500 hover:text-primary-600"
              >
                Upload another
              </button>
            </>
          ) : (
            <>
              <Upload className="w-8 h-8 mx-auto text-gray-400 mb-2" />
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                Drag & drop or{' '}
                <button
                  onClick={onButtonClick}
                  className="text-primary-500 hover:text-primary-600"
                >
                  browse
                </button>
              </p>
              <p className="text-xs text-gray-400">
                Supports: .tif, .tiff
              </p>
            </>
          )}
        </div>
        </div>
      )}
    </div>
  );
}
