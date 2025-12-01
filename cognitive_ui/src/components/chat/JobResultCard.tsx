import { Link } from 'react-router-dom';
import {
  CheckCircle,
  Clock,
  Loader2,
  XCircle,
  ExternalLink,
  Map,
  BarChart3,
  ArrowUpDown,
  Cpu,
  Grid3X3,
  Download,
} from 'lucide-react';
import type { Job, JobResultData } from '../../types';

interface JobResultCardProps {
  job: Job;
  resultData?: JobResultData;
  compact?: boolean;
}

// Helper to parse job result into structured data
export function parseJobResult(job: Job): JobResultData | undefined {
  if (!job.result) return undefined;

  // Handle conversational responses (no pipeline execution)
  if (job.result.intent === 'conversation') {
    return {
      summary: job.result.response || 'How can I help you with your geospatial analysis?',
      conversational: true,
    };
  }

  const execution = job.result.execution;
  const artifacts = execution?.artifacts || {};
  const resultData: JobResultData = {};

  // AI Summary
  if (artifacts.agent_result?.summary) {
    resultData.summary = artifacts.agent_result.summary;
  }

  // NDVI Map
  const ndviMap = artifacts.NDVIMap;
  if (ndviMap) {
    resultData.statistics = {
      type: 'ndvi',
      values: {
        mean: ndviMap.statistics?.mean?.toFixed(4) || 'N/A',
        std: ndviMap.statistics?.std?.toFixed(4) || 'N/A',
        min: ndviMap.statistics?.min?.toFixed(4) || 'N/A',
        max: ndviMap.statistics?.max?.toFixed(4) || 'N/A',
      },
    };

    if (ndviMap.visualizations?.ndvi_map) {
      resultData.visualizations = [
        {
          type: 'ndvi_map',
          label: 'NDVI Map',
          base64: ndviMap.visualizations.ndvi_map,
        },
      ];
    }
  }

  // Change Detection
  const changeMap = artifacts.ChangeMap;
  if (changeMap) {
    resultData.statistics = {
      type: 'change',
      values: {
        mean_change: changeMap.statistics?.mean_change?.toFixed(4) || 'N/A',
        std_change: changeMap.statistics?.std_change?.toFixed(4) || 'N/A',
      },
    };

    if (changeMap.classification) {
      resultData.classification = changeMap.classification;
    }

    const visualizations: JobResultData['visualizations'] = [];
    if (changeMap.visualizations?.change_map) {
      visualizations.push({
        type: 'change_map',
        label: 'Change Map',
        base64: changeMap.visualizations.change_map,
      });
    }
    if (changeMap.visualizations?.ndvi_before) {
      visualizations.push({
        type: 'ndvi_before',
        label: 'NDVI Before',
        base64: changeMap.visualizations.ndvi_before,
      });
    }
    if (changeMap.visualizations?.ndvi_after) {
      visualizations.push({
        type: 'ndvi_after',
        label: 'NDVI After',
        base64: changeMap.visualizations.ndvi_after,
      });
    }
    if (visualizations.length > 0) {
      resultData.visualizations = visualizations;
    }
  }

  // Statistics
  const statistics = artifacts.Statistics || artifacts.statistics;
  if (statistics && statistics.bands) {
    const bandStats = Object.entries(statistics.bands)[0];
    if (bandStats) {
      const [bandName, stats] = bandStats as [string, any];
      resultData.statistics = {
        type: 'statistics',
        values: {
          band: bandName,
          mean: stats.mean?.toFixed(4) || 'N/A',
          std: stats.std?.toFixed(4) || 'N/A',
          min: stats.min?.toFixed(4) || 'N/A',
          max: stats.max?.toFixed(4) || 'N/A',
        },
      };
    }
  }

  // Field Boundaries (Delineate-Anything)
  const fieldBoundaries = artifacts.FieldBoundaries;
  if (fieldBoundaries) {
    resultData.fieldBoundaries = {
      numFields: fieldBoundaries.num_fields || 0,
      totalAreaM2: fieldBoundaries.total_area_m2 || 0,
      outputPath: fieldBoundaries.output_path || '',
      crs: fieldBoundaries.crs || '',
    };

    // Add visualization if available
    if (fieldBoundaries.visualizations?.field_boundaries) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'field_boundaries',
        label: 'Detected Field Boundaries',
        base64: fieldBoundaries.visualizations.field_boundaries,
      });
    }
  }

  // Prithvi Features
  const features = artifacts.Features;
  if (features) {
    resultData.features = {
      dimensions: Array.isArray(features.features) ? features.features.length : 0,
      model: features.model || 'prithvi',
      version: features.version || 'v1.0',
    };
  }

  return resultData;
}

// Status badge component
function StatusBadge({ state }: { state: Job['state'] }) {
  const config = {
    pending: { icon: Clock, className: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300' },
    queued: { icon: Clock, className: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300' },
    running: { icon: Loader2, className: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' },
    completed: { icon: CheckCircle, className: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' },
    succeeded: { icon: CheckCircle, className: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' },
    failed: { icon: XCircle, className: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' },
    cancelled: { icon: XCircle, className: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300' },
  };

  const { icon: Icon, className } = config[state] || config.pending;
  const isAnimated = state === 'running';

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${className}`}>
      <Icon className={`w-3 h-3 ${isAnimated ? 'animate-spin' : ''}`} />
      {state}
    </span>
  );
}

// Result type icon
function ResultTypeIcon({ type }: { type: string }) {
  const icons: Record<string, typeof Map> = {
    ndvi: Map,
    change: ArrowUpDown,
    statistics: BarChart3,
    features: Cpu,
    field_boundaries: Grid3X3,
  };

  const Icon = icons[type] || BarChart3;
  return <Icon className="w-4 h-4" />;
}

export function JobResultCard({ job, resultData, compact = false }: JobResultCardProps) {
  const isComplete = job.state === 'completed' || job.state === 'succeeded';
  const isFailed = job.state === 'failed';
  const isRunning = job.state === 'running';

  // Parse result if not provided
  const data = resultData || (isComplete ? parseJobResult(job) : undefined);

  if (compact) {
    // Compact view for chat messages
    // For conversational responses, render just the text (no job card)
    if (data?.conversational) {
      return (
        <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
          {data.summary}
        </div>
      );
    }

    return (
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            {data?.statistics && <ResultTypeIcon type={data.statistics.type} />}
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Job {job.id.slice(0, 8)}
            </span>
            <StatusBadge state={job.state} />
          </div>
          <Link
            to={`/jobs/${job.id}`}
            className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
          >
            <ExternalLink className="w-4 h-4" />
          </Link>
        </div>

        {isRunning && (
          <div className="mt-2">
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                <div
                  className="bg-primary-500 h-1.5 rounded-full transition-all"
                  style={{ width: `${job.progress * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">{Math.round(job.progress * 100)}%</span>
            </div>
          </div>
        )}

        {isFailed && job.error && (
          <p className="mt-2 text-xs text-red-600 dark:text-red-400 truncate">
            {job.error}
          </p>
        )}

        {isComplete && data?.summary && (
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
            {data.summary}
          </p>
        )}
      </div>
    );
  }

  // Full view with visualizations
  // For conversational responses, render just the text (no job card)
  if (data?.conversational) {
    return (
      <div className="text-base text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
        {data.summary}
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {data?.statistics && <ResultTypeIcon type={data.statistics.type} />}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Analysis Result
              </h3>
              <p className="text-sm text-gray-500">Job {job.id}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge state={job.state} />
            <Link
              to={`/jobs/${job.id}`}
              className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400
                       bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              <ExternalLink className="w-4 h-4" />
              Details
            </Link>
          </div>
        </div>
      </div>

      {/* Progress (running) */}
      {isRunning && (
        <div className="p-4 bg-yellow-50 dark:bg-yellow-950 border-b border-yellow-200 dark:border-yellow-800">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-yellow-600 dark:text-yellow-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-yellow-700 dark:text-yellow-300">Processing...</span>
                <span className="text-sm text-yellow-600 dark:text-yellow-400">
                  {Math.round(job.progress * 100)}%
                </span>
              </div>
              <div className="bg-yellow-200 dark:bg-yellow-800 rounded-full h-2">
                <div
                  className="bg-yellow-500 h-2 rounded-full transition-all"
                  style={{ width: `${job.progress * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error (failed) */}
      {isFailed && job.error && (
        <div className="p-4 bg-red-50 dark:bg-red-950">
          <div className="flex items-start gap-2">
            <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700 dark:text-red-300">{job.error}</p>
          </div>
        </div>
      )}

      {/* Results (completed) */}
      {isComplete && data && (
        <div className="p-4 space-y-4">
          {/* AI Summary */}
          {data.summary && (
            <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg border border-purple-200 dark:border-purple-800">
              <p className="text-sm text-purple-800 dark:text-purple-200">{data.summary}</p>
            </div>
          )}

          {/* Statistics */}
          {data.statistics && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {data.statistics.type.toUpperCase()} Statistics
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(data.statistics.values).map(([key, value]) => (
                  <div key={key} className="bg-gray-50 dark:bg-gray-950 rounded-lg p-2">
                    <p className="text-xs text-gray-500 capitalize">{key.replace(/_/g, ' ')}</p>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Classification (Change Detection) */}
          {data.classification && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Classification
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                {Object.entries(data.classification).map(([key, val]) => {
                  const colorMap: Record<string, string> = {
                    severe_vegetation_loss: 'bg-red-100 dark:bg-red-950 border-red-200 dark:border-red-800 text-red-700 dark:text-red-300',
                    moderate_vegetation_loss: 'bg-orange-100 dark:bg-orange-950 border-orange-200 dark:border-orange-800 text-orange-700 dark:text-orange-300',
                    stable: 'bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300',
                    moderate_vegetation_gain: 'bg-lime-100 dark:bg-lime-950 border-lime-200 dark:border-lime-800 text-lime-700 dark:text-lime-300',
                    strong_vegetation_gain: 'bg-green-100 dark:bg-green-950 border-green-200 dark:border-green-800 text-green-700 dark:text-green-300',
                  };
                  return (
                    <div
                      key={key}
                      className={`rounded-lg p-2 border ${colorMap[key] || 'bg-gray-100 dark:bg-gray-800'}`}
                    >
                      <p className="text-xs capitalize">{key.replace(/_/g, ' ')}</p>
                      <p className="text-sm font-medium">{val.percentage?.toFixed(1)}%</p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Field Boundaries */}
          {data.fieldBoundaries && (
            <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2 mb-2">
                <Grid3X3 className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200">
                  Field Boundaries Detected
                </h4>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-blue-600 dark:text-blue-400">Fields Detected</p>
                  <p className="text-lg font-semibold text-blue-800 dark:text-blue-200">
                    {data.fieldBoundaries.numFields}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-blue-600 dark:text-blue-400">Total Area</p>
                  <p className="text-lg font-semibold text-blue-800 dark:text-blue-200">
                    {(data.fieldBoundaries.totalAreaM2 / 10000).toFixed(2)} ha
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Prithvi Features */}
          {data.features && (
            <div className="p-3 bg-indigo-50 dark:bg-indigo-950 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                <h4 className="text-sm font-medium text-indigo-800 dark:text-indigo-200">
                  Prithvi Features Extracted
                </h4>
              </div>
              <p className="text-sm text-indigo-700 dark:text-indigo-300">
                {data.features.dimensions}-dimensional feature vector from {data.features.model} {data.features.version}
              </p>
            </div>
          )}

          {/* Visualizations */}
          {data.visualizations && data.visualizations.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Visualizations
              </h4>
              <div className={`grid gap-3 ${data.visualizations.length === 1 ? 'grid-cols-1' : 'grid-cols-1 md:grid-cols-2'}`}>
                {data.visualizations.map((viz, i) => (
                  <div key={i} className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                    <div className="bg-gray-100 dark:bg-gray-800 px-3 py-2 flex items-center justify-between">
                      <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                        {viz.label}
                      </span>
                      <button
                        onClick={() => {
                          const link = document.createElement('a');
                          link.href = `data:image/png;base64,${viz.base64}`;
                          link.download = `${viz.type}_${job.id}.png`;
                          link.click();
                        }}
                        className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                        title="Download image"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                    </div>
                    <img
                      src={`data:image/png;base64,${viz.base64}`}
                      alt={viz.label}
                      className="w-full h-auto"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
