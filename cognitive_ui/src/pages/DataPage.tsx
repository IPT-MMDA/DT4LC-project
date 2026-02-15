import { useState, useRef, useEffect } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2, Download, Database } from 'lucide-react';
import { useUploadFile, useListFiles, GeoTIFFFile } from '../api/hooks/useUpload';
import { useAppStore } from '../store/useAppStore';

export function DataPage() {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadFile = useUploadFile();
  const { data: allFiles, isLoading: isLoadingFiles, refetch: refetchFiles } = useListFiles();
  const { addAttachment, uploadedAttachments } = useAppStore();

  // Refetch files after successful upload
  useEffect(() => {
    if (uploadFile.isSuccess) {
      refetchFiles();
    }
  }, [uploadFile.isSuccess, refetchFiles]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const handleFiles = async (files: FileList) => {
    const file = files[0];

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')) {
      alert('Please upload a .tif or .tiff file');
      return;
    }

    try {
      const result = await uploadFile.mutateAsync(file);

      // Add to global attachments store for use in chat
      addAttachment({
        id: result.id,
        filename: result.filename,
        path: result.path,
        mime_type: 'image/tiff',
      });
    } catch (error) {
      console.error('Upload error:', error);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  const addFileToChat = (file: GeoTIFFFile) => {
    // Check if already attached
    const alreadyAttached = uploadedAttachments.some(att => att.id === file.id);

    if (alreadyAttached) {
      alert(`"${file.filename}" is already attached to chat`);
      return;
    }

    addAttachment({
      id: file.id,
      filename: file.filename,
      path: file.path,
      mime_type: 'image/tiff',
    });

    // Show success feedback
    alert(`Added "${file.filename}" to chat. Go to Chat page to use it.`);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Data Management
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Upload and manage your geospatial data
        </p>
      </div>

      {/* Upload Zone */}
      <div
        className={`bg-white dark:bg-gray-900 rounded-lg border-2 border-dashed p-12 mb-8 transition-colors ${
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
        <div className="text-center">
          {uploadFile.isPending ? (
            <>
              <Loader2 className="w-12 h-12 mx-auto text-primary-500 mb-4 animate-spin" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Uploading...
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Processing your GeoTIFF file
              </p>
            </>
          ) : uploadFile.isError ? (
            <>
              <AlertCircle className="w-12 h-12 mx-auto text-red-500 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Upload Failed
              </h3>
              <p className="text-sm text-red-600 dark:text-red-400 mb-4">
                {uploadFile.error instanceof Error ? uploadFile.error.message : 'Failed to upload file'}
              </p>
              <button
                onClick={() => uploadFile.reset()}
                className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
              >
                Try Again
              </button>
            </>
          ) : (
            <>
              <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Upload GeoTIFF Data
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Drag and drop your .tif or .tiff files here, or click to browse
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".tif,.tiff"
                onChange={handleChange}
                className="hidden"
              />
              <button
                onClick={onButtonClick}
                className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
              >
                Select Files
              </button>
            </>
          )}
        </div>
      </div>

      {/* Available Files */}
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Available Files
          </h2>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {isLoadingFiles ? 'Loading...' : `${allFiles?.length || 0} file${allFiles?.length !== 1 ? 's' : ''}`}
            </span>
            <button
              onClick={() => refetchFiles()}
              className="text-sm text-primary-500 hover:text-primary-600"
            >
              Refresh
            </button>
          </div>
        </div>

        {isLoadingFiles ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
          </div>
        ) : allFiles && allFiles.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {allFiles.map((file) => (
              <div
                key={file.id}
                className="bg-gray-50 dark:bg-gray-950 rounded-lg border border-gray-200 dark:border-gray-800 p-4 relative group"
              >
                {/* Source badge */}
                <div className="absolute top-2 right-2 flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700">
                  {file.source === 'export' ? (
                    <>
                      <Download className="w-3 h-3 text-blue-500" />
                      <span className="text-blue-600 dark:text-blue-400">Export</span>
                    </>
                  ) : (
                    <>
                      <Upload className="w-3 h-3 text-green-500" />
                      <span className="text-green-600 dark:text-green-400">Upload</span>
                    </>
                  )}
                </div>

                {/* File Info */}
                <div className="space-y-2 mt-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0 pr-2">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate" title={file.filename}>
                        {file.filename}
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {file.size[0]} × {file.size[1]} pixels
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        {formatFileSize(file.size_bytes)}
                      </p>
                    </div>
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                  </div>

                  {file.crs && (
                    <div className="text-xs">
                      <span className="text-gray-600 dark:text-gray-400">CRS:</span>{' '}
                      <span className="text-gray-900 dark:text-white font-mono">
                        {file.crs}
                      </span>
                    </div>
                  )}

                  {file.bounds && (
                    <div className="text-xs">
                      <span className="text-gray-600 dark:text-gray-400">Bounds:</span>{' '}
                      <span className="text-gray-900 dark:text-white font-mono text-xs">
                        [{file.bounds[0].toFixed(2)}, {file.bounds[1].toFixed(2)}, {file.bounds[2].toFixed(2)}, {file.bounds[3].toFixed(2)}]
                      </span>
                    </div>
                  )}

                  <div className="text-xs text-gray-500 dark:text-gray-500">
                    {formatDate(file.modified)}
                  </div>

                  {/* Add to chat button */}
                  <button
                    onClick={() => addFileToChat(file)}
                    className={`w-full mt-2 px-3 py-1.5 text-sm rounded transition-colors ${
                      uploadedAttachments.some(att => att.id === file.id)
                        ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
                        : 'bg-primary-500 text-white hover:bg-primary-600'
                    }`}
                  >
                    {uploadedAttachments.some(att => att.id === file.id) ? '✓ Attached' : 'Add to Chat'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            <Database className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No files available</p>
            <p className="text-sm mt-1">Upload a GeoTIFF or export a layer from the map</p>
          </div>
        )}
      </div>
    </div>
  );
}
