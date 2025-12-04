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
  StopCircle,
  Snowflake,
} from 'lucide-react';
import type { Job, JobResultData } from '../../types';
import { useCancelJob } from '../../api/hooks/useJobs';

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

  // NDWI Map (Water Index)
  const ndwiMap = artifacts.NDWIMap;
  if (ndwiMap) {
    resultData.statistics = {
      type: 'ndwi',
      values: {
        mean: ndwiMap.statistics?.mean?.toFixed(4) || 'N/A',
        std: ndwiMap.statistics?.std?.toFixed(4) || 'N/A',
        min: ndwiMap.statistics?.min?.toFixed(4) || 'N/A',
        max: ndwiMap.statistics?.max?.toFixed(4) || 'N/A',
        water_coverage: ndwiMap.statistics?.water_percentage?.toFixed(2) + '%' || 'N/A',
      },
    };

    if (ndwiMap.visualizations?.ndwi_map) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'ndwi_map',
        label: 'NDWI Map',
        base64: ndwiMap.visualizations.ndwi_map,
      });
    }
    if (ndwiMap.visualizations?.water_mask) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'water_mask',
        label: 'Water Detection',
        base64: ndwiMap.visualizations.water_mask,
      });
    }
  }

  // NDSI Map (Snow Index)
  const ndsiMap = artifacts.NDSIMap;
  if (ndsiMap) {
    resultData.statistics = {
      type: 'ndsi',
      values: {
        mean: ndsiMap.statistics?.mean?.toFixed(4) || 'N/A',
        std: ndsiMap.statistics?.std?.toFixed(4) || 'N/A',
        min: ndsiMap.statistics?.min?.toFixed(4) || 'N/A',
        max: ndsiMap.statistics?.max?.toFixed(4) || 'N/A',
        snow_coverage: ndsiMap.statistics?.snow_coverage_percent?.toFixed(2) + '%' || 'N/A',
      },
    };

    if (ndsiMap.visualizations?.ndsi_map) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'ndsi_map',
        label: 'NDSI Map',
        base64: ndsiMap.visualizations.ndsi_map,
      });
    }
  }

  // LULC Classification (Land Use / Land Cover)
  const lulcMap = artifacts.LULCMap;
  if (lulcMap) {
    resultData.statistics = {
      type: 'lulc',
      values: {
        dominant_class: lulcMap.statistics?.dominant_class || 'N/A',
        valid_pixels: lulcMap.statistics?.valid_pixels?.toLocaleString() || 'N/A',
      },
    };

    // Add class distribution
    if (lulcMap.class_statistics) {
      resultData.classification = lulcMap.class_statistics;
    }

    if (lulcMap.visualizations?.classification_map) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'lulc_map',
        label: 'Land Cover Classification',
        base64: lulcMap.visualizations.classification_map,
      });
    }
    if (lulcMap.visualizations?.distribution_chart) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'lulc_distribution',
        label: 'Land Cover Distribution',
        base64: lulcMap.visualizations.distribution_chart,
      });
    }
  }

  // Snow Classification
  const snowClassification = artifacts.SnowClassification;
  if (snowClassification) {
    resultData.statistics = {
      type: 'snow',
      values: {
        snow_pixels: snowClassification.statistics?.snow_pixels?.toString() || 'N/A',
        total_pixels: snowClassification.statistics?.total_pixels?.toString() || 'N/A',
        snow_coverage: snowClassification.statistics?.snow_coverage_percent?.toFixed(2) + '%' || 'N/A',
      },
    };

    if (snowClassification.visualizations?.snow_classification) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'snow_classification',
        label: 'Snow Classification',
        base64: snowClassification.visualizations.snow_classification,
      });
    }
    if (snowClassification.visualizations?.criteria_analysis) {
      resultData.visualizations = resultData.visualizations || [];
      resultData.visualizations.push({
        type: 'criteria_analysis',
        label: 'Criteria Analysis',
        base64: snowClassification.visualizations.criteria_analysis,
      });
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

  // Prithvi Reconstruction (MAE outputs)
  const reconstruction = artifacts.Reconstruction;
  if (reconstruction) {
    resultData.reconstruction = {
      model: reconstruction.model || 'prithvi-eo-v1-100m',
      inputFile: reconstruction.input_file || '',
      outputDir: reconstruction.output_dir || '',
    };

    // Add visualizations from reconstruction output
    if (reconstruction.visualizations) {
      resultData.visualizations = resultData.visualizations || [];
      // Map visualization keys to user-friendly labels
      const labelMap: Record<string, string> = {
        original_rgb_t0: 'Original Image',
        masked_rgb_t0: 'Masked Image',
        predicted_rgb_t0: 'Reconstructed Image',
      };
      for (const [key, base64] of Object.entries(reconstruction.visualizations)) {
        if (typeof base64 === 'string') {
          resultData.visualizations.push({
            type: key,
            label: labelMap[key] || key.replace(/_/g, ' '),
            base64: base64,
          });
        }
      }
    }
  }

  return resultData;
}

// Status badge component
function StatusBadge({ status }: { status: Job['status'] }) {
  const config = {
    pending: { icon: Clock, className: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300' },
    running: { icon: Loader2, className: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' },
    completed: { icon: CheckCircle, className: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' },
    failed: { icon: XCircle, className: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' },
    cancelled: { icon: XCircle, className: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300' },
  };

  const { icon: Icon, className } = config[status] || config.pending;
  const isAnimated = status === 'running';

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${className}`}>
      <Icon className={`w-3 h-3 ${isAnimated ? 'animate-spin' : ''}`} />
      {status}
    </span>
  );
}

// Result type icon
function ResultTypeIcon({ type }: { type: string }) {
  const icons: Record<string, typeof Map> = {
    ndvi: Map,
    ndsi: Snowflake,
    snow: Snowflake,
    change: ArrowUpDown,
    statistics: BarChart3,
    features: Cpu,
    field_boundaries: Grid3X3,
  };

  const Icon = icons[type] || BarChart3;
  return <Icon className="w-4 h-4" />;
}

export function JobResultCard({ job, resultData, compact = false }: JobResultCardProps) {
  const isComplete = job.status === 'completed';
  const isFailed = job.status === 'failed';
  const isRunning = job.status === 'running';
  const isPending = job.status === 'pending';
  const canCancel = isRunning || isPending;

  // Cancel job mutation
  const cancelJob = useCancelJob();

  const handleCancel = () => {
    if (canCancel && !cancelJob.isPending) {
      cancelJob.mutate(job.id);
    }
  };

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
            <StatusBadge status={job.status} />
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
              <button
                onClick={handleCancel}
                disabled={cancelJob.isPending}
                className="text-red-500 hover:text-red-600 disabled:opacity-50"
                title="Cancel job"
              >
                <StopCircle className="w-4 h-4" />
              </button>
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
  // For conversational responses, render just the text ONCE (no job card)
  if (data?.conversational) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
        <p className="text-sm text-gray-900 dark:text-white whitespace-pre-wrap">
          {data.summary}
        </p>
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
            <StatusBadge status={job.status} />
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
                <div className="flex items-center gap-2">
                  <span className="text-sm text-yellow-600 dark:text-yellow-400">
                    {Math.round(job.progress * 100)}%
                  </span>
                  <button
                    onClick={handleCancel}
                    disabled={cancelJob.isPending}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-red-600 dark:text-red-400
                             bg-red-100 dark:bg-red-900 rounded hover:bg-red-200 dark:hover:bg-red-800
                             disabled:opacity-50 transition-colors"
                    title="Cancel job"
                  >
                    <StopCircle className="w-3 h-3" />
                    {cancelJob.isPending ? 'Cancelling...' : 'Cancel'}
                  </button>
                </div>
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

          {/* Prithvi Reconstruction */}
          {data.reconstruction && (
            <div className="p-3 bg-violet-50 dark:bg-violet-950 rounded-lg border border-violet-200 dark:border-violet-800">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-violet-600 dark:text-violet-400" />
                <h4 className="text-sm font-medium text-violet-800 dark:text-violet-200">
                  Prithvi MAE Reconstruction
                </h4>
              </div>
              <p className="text-sm text-violet-700 dark:text-violet-300">
                NASA/IBM foundation model analysis using {data.reconstruction.model}
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
