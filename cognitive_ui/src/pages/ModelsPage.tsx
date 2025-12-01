import { useQuery } from '@tanstack/react-query';
import { Loader2, CheckCircle, AlertCircle, Clock, Cpu, Map, ArrowUpDown, BarChart3, Grid3X3, ExternalLink } from 'lucide-react';
import apiClient from '../api/client';

interface ModelRequirement {
  model_id: string;
  name: string;
  description: string;
  author?: string;
  source_url?: string;
  available: boolean;
  missing_requirements: string[];
  gpu_required?: boolean;
  error?: string;
}

interface ModelsResponse {
  models: ModelRequirement[];
  count: number;
}

// Static model info for display
const MODEL_INFO: Record<string, { icon: typeof Cpu; color: string; keywords: string[] }> = {
  'prithvi': {
    icon: Cpu,
    color: 'indigo',
    keywords: ['features', 'embeddings', 'foundation model', 'temporal'],
  },
  'delineate-anything': {
    icon: Grid3X3,
    color: 'blue',
    keywords: ['field boundaries', 'agriculture', 'segmentation'],
  },
};

// Static algorithm info
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
    keywords: ['statistics', 'histogram', 'percentiles', 'analysis'],
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

function ModelCard({ model }: { model: ModelRequirement }) {
  const info = MODEL_INFO[model.model_id] || { icon: Cpu, color: 'gray', keywords: [] };
  const Icon = info.icon;
  const colorClasses: Record<string, string> = {
    indigo: 'bg-indigo-50 dark:bg-indigo-950 border-indigo-200 dark:border-indigo-800',
    blue: 'bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800',
    gray: 'bg-gray-50 dark:bg-gray-950 border-gray-200 dark:border-gray-800',
  };
  const iconColorClasses: Record<string, string> = {
    indigo: 'text-indigo-600 dark:text-indigo-400',
    blue: 'text-blue-600 dark:text-blue-400',
    gray: 'text-gray-600 dark:text-gray-400',
  };

  return (
    <div className={`rounded-lg border p-6 ${colorClasses[info.color] || colorClasses.gray}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${colorClasses[info.color] || colorClasses.gray}`}>
            <Icon className={`w-5 h-5 ${iconColorClasses[info.color] || iconColorClasses.gray}`} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {model.name}
            </h3>
            {model.author && (
              <p className="text-xs text-gray-500 dark:text-gray-500">
                by {model.author}
              </p>
            )}
          </div>
        </div>
        <StatusBadge available={model.available} missing={model.missing_requirements} />
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
          View on {model.source_url.includes('huggingface') ? 'HuggingFace' : 'GitHub'}
        </a>
      )}

      {/* Keywords */}
      <div className="flex flex-wrap gap-1.5">
        {info.keywords.map((kw) => (
          <span
            key={kw}
            className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 text-xs rounded"
          >
            {kw}
          </span>
        ))}
        {model.gpu_required && (
          <span className="px-2 py-0.5 bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300 text-xs rounded">
            GPU recommended
          </span>
        )}
      </div>

      {/* Missing requirements */}
      {!model.available && model.missing_requirements && model.missing_requirements.length > 0 && (
        <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg border border-yellow-200 dark:border-yellow-800">
          <p className="text-xs font-medium text-yellow-800 dark:text-yellow-200 mb-1">
            Missing requirements:
          </p>
          <ul className="text-xs text-yellow-700 dark:text-yellow-300 space-y-0.5">
            {model.missing_requirements.map((req) => (
              <li key={req}>- {req}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Error */}
      {model.error && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800">
          <p className="text-xs text-red-700 dark:text-red-300">{model.error}</p>
        </div>
      )}
    </div>
  );
}

function AlgorithmCard({ algorithm }: { algorithm: typeof ALGORITHMS[0] }) {
  const Icon = algorithm.icon;
  const colorClasses: Record<string, string> = {
    green: 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800',
    purple: 'bg-purple-50 dark:bg-purple-950 border-purple-200 dark:border-purple-800',
    cyan: 'bg-cyan-50 dark:bg-cyan-950 border-cyan-200 dark:border-cyan-800',
  };
  const iconColorClasses: Record<string, string> = {
    green: 'text-green-600 dark:text-green-400',
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

export function ModelsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['models'],
    queryFn: () => apiClient.get<ModelsResponse>('/v1/models'),
    staleTime: 60000, // 1 minute
  });

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
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Machine Learning Models
        </h2>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
          </div>
        ) : error ? (
          <div className="bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800 p-6">
            <div className="flex items-center gap-2 text-red-700 dark:text-red-300">
              <AlertCircle className="w-5 h-5" />
              <p>Failed to load models: {(error as Error).message}</p>
            </div>
          </div>
        ) : data?.models && data.models.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {data.models.map((model: ModelRequirement) => (
              <ModelCard key={model.model_id} model={model} />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Fallback static cards when API not available */}
            <div className="bg-indigo-50 dark:bg-indigo-950 rounded-lg border border-indigo-200 dark:border-indigo-800 p-6">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-indigo-100 dark:bg-indigo-900">
                    <Cpu className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Prithvi Model
                  </h3>
                </div>
                <span className="inline-flex items-center gap-1 px-2 py-1 bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300 text-xs rounded">
                  <Clock className="w-3 h-3" />
                  Setup Required
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                IBM/NASA foundation model for Earth observation. Extracts temporal features
                and embeddings from HLS satellite data.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {['features', 'embeddings', 'foundation model', 'temporal'].map((kw) => (
                  <span
                    key={kw}
                    className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 text-xs rounded"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800 p-6">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900">
                    <Grid3X3 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Delineate-Anything
                  </h3>
                </div>
                <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300 text-xs rounded">
                  <CheckCircle className="w-3 h-3" />
                  Available
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Agricultural field boundary detection using SAM-based segmentation.
                Outputs GeoPackage with field polygons and area statistics.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {['field boundaries', 'agriculture', 'segmentation', 'polygons'].map((kw) => (
                  <span
                    key={kw}
                    className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 text-xs rounded"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
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
              <li>- "Analyze vegetation health and detect changes" (Change Detection)</li>
              <li>- "Get statistics for all bands in my raster" (Statistical Analysis)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
