import { CheckCircle, AlertCircle, BarChart3, TrendingUp, Map, Download, ArrowUpDown, Grid3X3, Cpu } from 'lucide-react';

interface JobResultProps {
  result: any;
}

// Helper to download base64 image
function downloadBase64Image(base64: string, filename: string) {
  const link = document.createElement('a');
  link.href = `data:image/png;base64,${base64}`;
  link.download = filename;
  link.click();
}

export function JobResult({ result }: JobResultProps) {
  if (!result) return null;

  // Parse the actual backend structure
  const execution = result.execution;
  const artifacts = execution?.artifacts || {};

  // Statistics can be at top level or nested
  const statistics = artifacts.Statistics || artifacts.statistics;

  // Change detection results
  const changeMap = artifacts.ChangeMap;

  // NDVI Map results
  const ndviMap = artifacts.NDVIMap;

  // Field Boundaries (Delineate-Anything)
  const fieldBoundaries = artifacts.FieldBoundaries;

  // Prithvi Reconstruction (MAE outputs)
  const reconstruction = artifacts.Reconstruction;

  // AI summary is in agent_result
  const aiSummary = artifacts.agent_result?.summary;

  return (
    <div className="space-y-6">
      {/* Execution Status */}
      {execution && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Execution Complete
            </h3>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Pipeline executed successfully in{' '}
            {execution.duration_seconds?.toFixed(2) || 'N/A'} seconds
          </p>
        </div>
      )}

      {/* Field Boundaries (Delineate-Anything) Results */}
      {fieldBoundaries && (
        <div className="bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Grid3X3 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Field Boundary Detection
              </h3>
            </div>
            <div className="flex items-center gap-2">
              {fieldBoundaries.visualizations?.field_boundaries && (
                <button
                  onClick={() => downloadBase64Image(fieldBoundaries.visualizations.field_boundaries, 'field_boundaries.png')}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400
                             bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  <Download className="w-4 h-4" />
                  Download PNG
                </button>
              )}
              {fieldBoundaries.output_path && (
                <a
                  href={`/v1/download?path=${encodeURIComponent(fieldBoundaries.output_path)}`}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-blue-600 dark:text-blue-400
                             bg-blue-100 dark:bg-blue-900 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-800"
                  download
                >
                  <Download className="w-4 h-4" />
                  Download GeoPackage
                </a>
              )}
            </div>
          </div>

          {/* Field Boundaries Visualization */}
          {fieldBoundaries.visualizations?.field_boundaries && (
            <div className="mb-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Detected Field Boundaries
              </p>
              <div className="rounded-lg overflow-hidden border border-blue-200 dark:border-blue-700">
                <img
                  src={`data:image/png;base64,${fieldBoundaries.visualizations.field_boundaries}`}
                  alt="Field Boundaries"
                  className="w-full h-auto"
                />
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-900 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <p className="text-xs text-blue-600 dark:text-blue-400 mb-1">Fields Detected</p>
              <p className="text-2xl font-bold text-blue-800 dark:text-blue-200">
                {fieldBoundaries.num_fields || 0}
              </p>
            </div>
            <div className="bg-white dark:bg-gray-900 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <p className="text-xs text-blue-600 dark:text-blue-400 mb-1">Total Area</p>
              <p className="text-2xl font-bold text-blue-800 dark:text-blue-200">
                {((fieldBoundaries.total_area_m2 || 0) / 10000).toFixed(2)} ha
              </p>
            </div>
            <div className="bg-white dark:bg-gray-900 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <p className="text-xs text-blue-600 dark:text-blue-400 mb-1">CRS</p>
              <p className="text-sm font-medium text-blue-800 dark:text-blue-200 truncate">
                {fieldBoundaries.crs || 'Unknown'}
              </p>
            </div>
            <div className="bg-white dark:bg-gray-900 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
              <p className="text-xs text-blue-600 dark:text-blue-400 mb-1">Model</p>
              <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                {fieldBoundaries.model || 'Delineate-Anything'}
              </p>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              Field boundaries have been extracted and saved to a GeoPackage file.
              You can download and open it in QGIS or any GIS software.
            </p>
          </div>
        </div>
      )}

      {/* Prithvi Reconstruction Results */}
      {reconstruction && (
        <div className="bg-violet-50 dark:bg-violet-950 rounded-lg border border-violet-200 dark:border-violet-800 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-5 h-5 text-violet-600 dark:text-violet-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Prithvi MAE Reconstruction
            </h3>
          </div>

          <p className="text-sm text-violet-700 dark:text-violet-300 mb-4">
            NASA/IBM foundation model analysis using {reconstruction.model || 'prithvi-eo-v1-100m'}
          </p>

          {/* Reconstruction Visualizations */}
          {reconstruction.visualizations && Object.keys(reconstruction.visualizations).length > 0 && (
            <div className="space-y-4">
              <p className="text-sm font-medium text-violet-700 dark:text-violet-300">
                Reconstruction Output
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(reconstruction.visualizations).map(([key, base64]) => {
                  const labelMap: Record<string, string> = {
                    original_rgb_t0: 'Original Image',
                    masked_rgb_t0: 'Masked Image',
                    predicted_rgb_t0: 'Reconstructed Image',
                  };
                  const label = labelMap[key] || key.replace(/_/g, ' ');
                  return (
                    <div key={key} className="rounded-lg overflow-hidden border border-violet-200 dark:border-violet-700">
                      <div className="bg-violet-100 dark:bg-violet-900 px-3 py-2 flex items-center justify-between">
                        <span className="text-xs font-medium text-violet-700 dark:text-violet-300">
                          {label}
                        </span>
                        <button
                          onClick={() => downloadBase64Image(base64 as string, `${key}.png`)}
                          className="text-violet-500 hover:text-violet-700 dark:hover:text-violet-300"
                          title="Download"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                      <img
                        src={`data:image/png;base64,${base64}`}
                        alt={label}
                        className="w-full h-auto"
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Change Detection Results */}
      {changeMap && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <ArrowUpDown className="w-5 h-5 text-primary-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Vegetation Change Detection
              </h3>
            </div>
            {changeMap.visualizations?.change_map && (
              <button
                onClick={() => downloadBase64Image(changeMap.visualizations.change_map, 'change_map.png')}
                className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400
                           bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            )}
          </div>

          {/* Visualizations */}
          {changeMap.visualizations && (
            <div className="space-y-4">
              {/* Main Change Map */}
              {changeMap.visualizations.change_map && (
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Change Map (Red: Loss, Green: Gain)
                  </p>
                  <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                    <img
                      src={`data:image/png;base64,${changeMap.visualizations.change_map}`}
                      alt="Vegetation Change Map"
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              )}

              {/* Before/After NDVI */}
              {(changeMap.visualizations.ndvi_before || changeMap.visualizations.ndvi_after) && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {changeMap.visualizations.ndvi_before && (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                          NDVI - Before
                        </p>
                        <button
                          onClick={() => downloadBase64Image(changeMap.visualizations.ndvi_before, 'ndvi_before.png')}
                          className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                          title="Download"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                        <img
                          src={`data:image/png;base64,${changeMap.visualizations.ndvi_before}`}
                          alt="NDVI Before"
                          className="w-full h-auto"
                        />
                      </div>
                    </div>
                  )}
                  {changeMap.visualizations.ndvi_after && (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                          NDVI - After
                        </p>
                        <button
                          onClick={() => downloadBase64Image(changeMap.visualizations.ndvi_after, 'ndvi_after.png')}
                          className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                          title="Download"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                        <img
                          src={`data:image/png;base64,${changeMap.visualizations.ndvi_after}`}
                          alt="NDVI After"
                          className="w-full h-auto"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Classification Stats */}
          {changeMap.classification && (
            <div className="mt-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Change Classification
              </p>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-red-50 dark:bg-red-950 rounded-lg p-3 border border-red-200 dark:border-red-800">
                  <p className="text-xs text-red-600 dark:text-red-400">Severe Loss</p>
                  <p className="text-lg font-semibold text-red-700 dark:text-red-300">
                    {changeMap.classification.severe_vegetation_loss?.percentage?.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-950 rounded-lg p-3 border border-orange-200 dark:border-orange-800">
                  <p className="text-xs text-orange-600 dark:text-orange-400">Moderate Loss</p>
                  <p className="text-lg font-semibold text-orange-700 dark:text-orange-300">
                    {changeMap.classification.moderate_vegetation_loss?.percentage?.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                  <p className="text-xs text-gray-600 dark:text-gray-400">Stable</p>
                  <p className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                    {changeMap.classification.stable?.percentage?.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-lime-50 dark:bg-lime-950 rounded-lg p-3 border border-lime-200 dark:border-lime-800">
                  <p className="text-xs text-lime-600 dark:text-lime-400">Moderate Gain</p>
                  <p className="text-lg font-semibold text-lime-700 dark:text-lime-300">
                    {changeMap.classification.moderate_vegetation_gain?.percentage?.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-green-50 dark:bg-green-950 rounded-lg p-3 border border-green-200 dark:border-green-800">
                  <p className="text-xs text-green-600 dark:text-green-400">Strong Gain</p>
                  <p className="text-lg font-semibold text-green-700 dark:text-green-300">
                    {changeMap.classification.strong_vegetation_gain?.percentage?.toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Change Statistics */}
          {changeMap.statistics && (
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Mean Change</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {changeMap.statistics.mean_change?.toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Std Dev</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {changeMap.statistics.std_change?.toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Min Change</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {changeMap.statistics.min_change?.toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Max Change</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {changeMap.statistics.max_change?.toFixed(4)}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* NDVI Map Results (single image analysis) */}
      {ndviMap && ndviMap.visualizations && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Map className="w-5 h-5 text-green-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                NDVI Analysis
              </h3>
            </div>
            {ndviMap.visualizations.ndvi_map && (
              <button
                onClick={() => downloadBase64Image(ndviMap.visualizations.ndvi_map, 'ndvi_map.png')}
                className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400
                           bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            )}
          </div>

          {ndviMap.visualizations.ndvi_map && (
            <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
              <img
                src={`data:image/png;base64,${ndviMap.visualizations.ndvi_map}`}
                alt="NDVI Map"
                className="w-full h-auto"
              />
            </div>
          )}

          {/* NDVI Statistics */}
          {ndviMap.statistics && (
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Mean NDVI</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {ndviMap.statistics.mean?.toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Std Dev</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {ndviMap.statistics.std?.toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Min</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {ndviMap.statistics.min?.toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-600 dark:text-gray-400">Max</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {ndviMap.statistics.max?.toFixed(4)}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Statistics Results */}
      {statistics && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-primary-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Statistical Analysis
            </h3>
          </div>

          {/* Band Statistics - handle both bands and band_statistics */}
          {(statistics.bands || statistics.band_statistics) && (
            <div className="space-y-4">
              {Object.entries(statistics.bands || statistics.band_statistics).map(([band, stats]: [string, any]) => (
                <div key={band} className="border-l-4 border-primary-500 pl-4">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                    {band}
                  </h4>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {/* Basic Stats */}
                    <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                      <p className="text-xs text-gray-600 dark:text-gray-400">Mean</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {stats.mean?.toFixed(4) || 'N/A'}
                      </p>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                      <p className="text-xs text-gray-600 dark:text-gray-400">Std Dev</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {stats.std?.toFixed(4) || 'N/A'}
                      </p>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                      <p className="text-xs text-gray-600 dark:text-gray-400">Min</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {stats.min?.toFixed(4) || 'N/A'}
                      </p>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-950 rounded-lg p-3">
                      <p className="text-xs text-gray-600 dark:text-gray-400">Max</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {stats.max?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                  </div>

                  {/* Percentiles */}
                  {stats.percentiles && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        Percentiles
                      </p>
                      <div className="grid grid-cols-5 gap-2">
                        {Object.entries(stats.percentiles).map(([p, val]: [string, any]) => (
                          <div
                            key={p}
                            className="bg-gray-50 dark:bg-gray-950 rounded px-2 py-1 text-center"
                          >
                            <p className="text-xs text-gray-600 dark:text-gray-400">
                              {p}
                            </p>
                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                              {val?.toFixed(2)}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Histogram */}
                  {stats.histogram && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        Distribution (10 bins)
                      </p>
                      <div className="flex items-end gap-1 h-20">
                        {stats.histogram.counts?.map((count: number, i: number) => {
                          const maxCount = Math.max(...stats.histogram.counts);
                          const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
                          return (
                            <div
                              key={i}
                              className="flex-1 bg-primary-500 rounded-t"
                              style={{ height: `${height}%` }}
                              title={`Bin ${i + 1}: ${count} pixels`}
                            />
                          );
                        })}
                      </div>
                      <div className="flex justify-between mt-1">
                        <span className="text-xs text-gray-600 dark:text-gray-400">
                          {stats.histogram.bin_edges?.[0]?.toFixed(2)}
                        </span>
                        <span className="text-xs text-gray-600 dark:text-gray-400">
                          {stats.histogram.bin_edges?.[
                            stats.histogram.bin_edges.length - 1
                          ]?.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Summary Text (if available) */}
          {statistics.summary && (
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-900 dark:text-blue-100">
                {statistics.summary}
              </p>
            </div>
          )}
        </div>
      )}

      {/* LLM Summary (if available) */}
      {aiSummary && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-purple-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              AI Analysis Summary
            </h3>
          </div>
          <div className="prose dark:prose-invert max-w-none">
            <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
              {aiSummary}
            </p>
          </div>
        </div>
      )}

      {/* LLM Summary Failed Notice */}
      {!aiSummary && execution && (
        <div className="bg-yellow-50 dark:bg-yellow-950 rounded-lg border border-yellow-200 dark:border-yellow-800 p-4">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-yellow-900 dark:text-yellow-100">
                AI Summary Unavailable
              </p>
              <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                The analysis completed successfully, but the AI summary could not be
                generated (LLM quota exceeded). The statistical results above are
                complete and accurate.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Raw Artifacts (Debug - can be removed) */}
      {artifacts && Object.keys(artifacts).length > 0 && (
        <details className="bg-gray-50 dark:bg-gray-950 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <summary className="text-sm font-medium text-gray-700 dark:text-gray-300 cursor-pointer">
            View Raw Data (Debug)
          </summary>
          <pre className="mt-2 text-xs text-gray-600 dark:text-gray-400 overflow-auto max-h-96">
            {JSON.stringify(artifacts, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
