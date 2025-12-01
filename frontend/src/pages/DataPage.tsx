import { useState, useRef } from 'react';
import { Upload, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { useUploadFile } from '../api/hooks/useUpload';
import { useAppStore } from '../store/useAppStore';

interface UploadedFile {
  filename: string;
  size: [number, number];
  crs: string | null;
  bounds: [number, number, number, number];
  preview_png_base64: string;
}

export function DataPage() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadFile = useUploadFile();
  const { addAttachment } = useAppStore();

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
      setUploadedFiles((prev) => [...prev, result]);

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

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
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

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Uploaded Files
            </h2>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {uploadedFiles.length} file{uploadedFiles.length !== 1 ? 's' : ''}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {uploadedFiles.map((file, index) => (
              <div
                key={index}
                className="bg-gray-50 dark:bg-gray-950 rounded-lg border border-gray-200 dark:border-gray-800 p-4 relative group"
              >
                <button
                  onClick={() => removeFile(index)}
                  className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="w-4 h-4" />
                </button>

                {/* Preview */}
                {file.preview_png_base64 && (
                  <div className="mb-3 rounded overflow-hidden bg-gray-200 dark:bg-gray-800">
                    <img
                      src={`data:image/png;base64,${file.preview_png_base64}`}
                      alt={file.filename}
                      className="w-full h-32 object-cover"
                    />
                  </div>
                )}

                {/* File Info */}
                <div className="space-y-2">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {file.filename}
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {file.size[0]} × {file.size[1]} pixels
                      </p>
                    </div>
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 ml-2" />
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
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
