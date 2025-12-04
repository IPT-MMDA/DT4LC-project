import { useQuery } from '@tanstack/react-query';
import { Loader2, CheckCircle, AlertCircle, Clock, Cpu, Map, ArrowUpDown, BarChart3, ExternalLink, Download, Trash2, XCircle, Snowflake } from 'lucide-react';
import apiClient from '../api/client';
import { useMLModels, useDownloadModel, useDeleteModel, useCancelDownload, type MLModel } from '../api/hooks/useMLModels';

interface HostedModel {
  model_id: string;
  name: string;
  description: string;
  author?: string;
  source_url?: string;
  available: boolean;
  missing_requirements: string[];
  gpu_required?: boolean;
  error?: string;
  integration_type?: string;
  integration_status?: string;
  keywords?: string[];
  hosting?: string;
  team?: string;
}

interface ModelsResponse {
  models: HostedModel[];
  count: number;
}

// Static algorithm info - synced with dta/registry.yaml
const ALGORITHMS = [
  {
    id: 'ndvi',
    name: 'NDVI Calculation',
    description: 'Normalized Difference Vegetation Index for vegetation health analysis. Uses NIR and Red bands to calculate vegetation density.',
    icon: Map,
    color: 'green',
    available: true,
    keywords: ['vegetation', 'health', 'greenness', 'ndvi'],
  },
  {
    id: 'ndsi',
    name: 'NDSI Snow Index',
    description: 'Normalized Difference Snow Index for snow and glacier detection. Computed as (Green - SWIR) / (Green + SWIR). Values above 0.42 typically indicate snow/ice.',
    icon: Snowflake,
    color: 'blue',
    available: true,
    keywords: ['ndsi', 'snow', 'ice', 'glacier', 'cryosphere', 'alpine'],
  },
  {
    id: 'snow-classifier',
    name: 'Snow Classification',
    description: 'Multi-criteria snow classification combining NDSI (>=0.4), NDVI (~0.1), and brightness (>0.3) thresholds for robust snow detection.',
    icon: Snowflake,
    color: 'sky',
    available: true,
    keywords: ['snow', 'ice', 'glacier', 'multi-criteria', 'robust'],
  },
  {
    id: 'change-detection',
    name: 'Change Detection',
    description: 'Detect and quantify changes in land cover between two time periods. Compares NDVI values to identify vegetation loss or gain.',
    icon: ArrowUpDown,
    color: 'purple',
    available: true,
    keywords: ['change', 'before', 'after', 'difference', 'temporal'],
  },
  {
    id: 'statistics',
    name: 'Statistical Analysis',
    description: 'Comprehensive raster statistics including mean, std, percentiles, and histograms for all bands.',
    icon: BarChart3,
    color: 'cyan',
    available: true,
    keywords: ['statistics', 'histogram', 'percentiles', 'distribution'],
  },
];

function StatusBadge({ available, missing }: { available: boolean; missing?: string[] }) {
  if (available) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300 text-xs rounded">
        <CheckCircle className="w-3 h-3" />
        Available
      </span>
    );
  }

  if (missing && missing.length > 0) {
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-1 bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300 text-xs rounded"
        title={`Missing: ${missing.join(', ')}`}
      >
        <Clock className="w-3 h-3" />
        Setup Required
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300 text-xs rounded">
      <AlertCircle className="w-3 h-3" />
      Unavailable
    </span>
  );
}

function HostedModelCard({ model }: { model: HostedModel }) {
  const getHostingLabel = () => {
    if (model.integration_type === 'google-earth-engine') return 'Google Earth Engine';
    if (model.hosting === 'huggingface') return 'HuggingFace Spaces';
    return model.hosting || 'External';
  };

  return (
    <div className="rounded-lg border p-6 bg-amber-50 dark:bg-amber-950 border-amber-200 dark:border-amber-800">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-amber-100 dark:bg-amber-900">
            <Cpu className="w-5 h-5 text-amber-600 dark:text-amber-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {model.name}
            </h3>
            {model.author && (
              <p className="text-xs text-gray-500 dark:text-gray-500">
                by {model.author}{model.team && ` (${model.team})`}
              </p>
            )}
          </div>
        </div>
        <span className="inline-flex items-center gap-1 px-2 py-1 bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300 text-xs rounded">
          <Clock className="w-3 h-3" />
          {model.integration_status === 'planned' ? 'Coming Soon' : model.integration_status}
        </span>
      </div>

      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {model.description}
      </p>

      {/* Source link */}
      {model.source_url && (
        <a
          href={model.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-xs text-primary-600 dark:text-primary-400 hover:underline mb-3"
        >
          <ExternalLink className="w-3 h-3" />
          View Source
        </a>
      )}

      {/* Hosting badge */}
      <div className="flex flex-wrap gap-1.5 mb-2">
        <span className="px-2 py-0.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded font-medium">
          {getHostingLabel()}
        </span>
      </div>

      {/* Keywords */}
      {model.keywords && model.keywords.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {model.keywords.slice(0, 5).map((kw) => (
            <span
              key={kw}
              className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 text-xs rounded"
            >
              {kw}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function AlgorithmCard({ algorithm }: { algorithm: typeof ALGORITHMS[0] }) {
  const Icon = algorithm.icon;
  const colorClasses: Record<string, string> = {
    green: 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800',
    blue: 'bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800',
    sky: 'bg-sky-50 dark:bg-sky-950 border-sky-200 dark:border-sky-800',
    purple: 'bg-purple-50 dark:bg-purple-950 border-purple-200 dark:border-purple-800',
    cyan: 'bg-cyan-50 dark:bg-cyan-950 border-cyan-200 dark:border-cyan-800',
  };
  const iconColorClasses: Record<string, string> = {
    green: 'text-green-600 dark:text-green-400',
    blue: 'text-blue-600 dark:text-blue-400',
    sky: 'text-sky-600 dark:text-sky-400',
    purple: 'text-purple-600 dark:text-purple-400',
    cyan: 'text-cyan-600 dark:text-cyan-400',
  };

  return (
    <div className={`rounded-lg border p-6 ${colorClasses[algorithm.color]}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${colorClasses[algorithm.color]}`}>
            <Icon className={`w-5 h-5 ${iconColorClasses[algorithm.color]}`} />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {algorithm.name}
          </h3>
        </div>
        <StatusBadge available={algorithm.available} />
      </div>

      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {algorithm.description}
      </p>

      {/* Keywords */}
      <div className="flex flex-wrap gap-1.5">
        {algorithm.keywords.map((kw) => (
          <span
            key={kw}
            className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 text-xs rounded"
          >
            {kw}
          </span>
        ))}
      </div>
    </div>
  );
}

// ML Model Card with download/delete functionality
function MLModelCard({ model }: { model: MLModel }) {
  const downloadMutation = useDownloadModel();
  const deleteMutation = useDeleteModel();
  const cancelMutation = useCancelDownload();

  const isDownloading = model.status === 'downloading';
  const isAvailable = model.status === 'available';
  const progress = model.download_progress;

  const handleDownload = () => {
    downloadMutation.mutate(model.id);
  };

  const handleDelete = () => {
    if (confirm(`Delete ${model.name}? This will free ${model.size_mb} MB.`)) {
      deleteMutation.mutate(model.id);
    }
  };

  const handleCancel = () => {
    cancelMutation.mutate(model.id);
  };

  const colorClasses = isAvailable
    ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800'
    : isDownloading
    ? 'bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800'
    : 'bg-gray-50 dark:bg-gray-950 border-gray-200 dark:border-gray-800';

  return (
    <div className={`rounded-lg border p-6 ${colorClasses}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-100 dark:bg-indigo-900">
            <Cpu className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {model.name}
            </h3>
            <p className="text-xs text-gray-500">{model.size_mb} MB</p>
          </div>
        </div>

        {/* Status badge */}
        {isAvailable ? (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300 text-xs rounded">
            <CheckCircle className="w-3 h-3" />
            Installed
          </span>
        ) : isDownloading ? (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300 text-xs rounded">
            <Loader2 className="w-3 h-3 animate-spin" />
            Downloading
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300 text-xs rounded">
            <Download className="w-3 h-3" />
            Not Installed
          </span>
        )}
      </div>

      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {model.description}
      </p>

      {/* Download progress bar */}
      {isDownloading && progress && (
        <div className="mb-4">
          <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
            <span>{progress.percent.toFixed(1)}% ({progress.downloaded_mb.toFixed(0)}/{progress.total_mb.toFixed(0)} MB)</span>
            {progress.eta_seconds && <span>~{progress.eta_seconds}s remaining</span>}
          </div>
          <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
          {progress.speed_mbps > 0 && (
            <p className="text-xs text-gray-500 mt-1">{progress.speed_mbps.toFixed(1)} MB/s</p>
          )}
        </div>
      )}

      {/* License info */}
      <a
        href={model.license_url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 text-xs text-primary-600 dark:text-primary-400 hover:underline mb-4"
      >
        <ExternalLink className="w-3 h-3" />
        {model.license} License
      </a>

      {/* Action buttons */}
      <div className="flex gap-2 mt-2">
        {!isAvailable && !isDownloading && (
          <button
            onClick={handleDownload}
            disabled={downloadMutation.isPending}
            className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {downloadMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            Download
          </button>
        )}

        {isDownloading && (
          <button
            onClick={handleCancel}
            disabled={cancelMutation.isPending}
            className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <XCircle className="w-4 h-4" />
            Cancel
          </button>
        )}

        {isAvailable && (
          <button
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {deleteMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Trash2 className="w-4 h-4" />
            )}
            Delete
          </button>
        )}
      </div>

      {/* Error display */}
      {model.error && (
        <div className="mt-3 p-2 bg-red-50 dark:bg-red-950 rounded border border-red-200 dark:border-red-800">
          <p className="text-xs text-red-700 dark:text-red-300">{model.error}</p>
        </div>
      )}
    </div>
  );
}

// ML Models section component - displays both local and hosted models
function MLModelsSection({ hostedModels }: { hostedModels: HostedModel[] }) {
  const { data, isLoading, error } = useMLModels();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800 p-6">
        <div className="flex items-center gap-2 text-red-700 dark:text-red-300">
          <AlertCircle className="w-5 h-5" />
          <p>Failed to load ML models: {(error as Error).message}</p>
        </div>
      </div>
    );
  }

  const localModels = data?.models || [];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {localModels.map((model) => (
        <MLModelCard key={model.id} model={model} />
      ))}
      {hostedModels.map((model) => (
        <HostedModelCard key={model.model_id} model={model} />
      ))}
    </div>
  );
}

export function ModelsPage() {
  const { data } = useQuery({
    queryKey: ['models'],
    queryFn: () => apiClient.get<ModelsResponse>('/v1/models'),
    staleTime: 60000, // 1 minute
  });

  // Filter hosted models (external platforms like GEE, HuggingFace Spaces)
  const hostedModels = data?.models?.filter((m) => m.integration_type) || [];

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Models & Capabilities
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Explore available models and algorithms for geospatial analysis
        </p>
      </div>

      {/* Algorithms Section */}
      <div className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Algorithms
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {ALGORITHMS.map((algo) => (
            <AlgorithmCard key={algo.id} algorithm={algo} />
          ))}
        </div>
      </div>

      {/* ML Models Section */}
      <div className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          ML Models
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Machine learning models for advanced geospatial analysis.
        </p>
        <MLModelsSection hostedModels={hostedModels} />
      </div>

      {/* Usage Instructions */}
      <div className="mt-10 bg-gray-50 dark:bg-gray-950 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          How to Use
        </h2>
        <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
          <p>
            <strong>1. Upload Data:</strong> Go to the Data page and upload your GeoTIFF satellite imagery.
          </p>
          <p>
            <strong>2. Start Analysis:</strong> In the Chat page, describe what analysis you want to perform.
            The system will automatically select the appropriate model or algorithm.
          </p>
          <p>
            <strong>3. View Results:</strong> Results will appear in the chat with visualizations and statistics.
            Click on job links to see detailed processing information.
          </p>
          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-blue-800 dark:text-blue-200">
              <strong>Example prompts:</strong>
            </p>
            <ul className="mt-2 space-y-1 text-blue-700 dark:text-blue-300">
              <li>- "Extract field boundaries from my satellite image" (Delineate-Anything)</li>
              <li>- "Detect agricultural parcels in this area" (Delineate-Anything)</li>
              <li>- "Calculate NDVI for my uploaded image" (NDVI Algorithm)</li>
              <li>- "Detect snow and ice coverage in this image" (NDSI / Snow Classifier)</li>
              <li>- "Analyze glacier extent in my satellite imagery" (Snow Classification)</li>
              <li>- "Analyze vegetation health and detect changes" (Change Detection)</li>
              <li>- "Get statistics for all bands in my raster" (Statistical Analysis)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
