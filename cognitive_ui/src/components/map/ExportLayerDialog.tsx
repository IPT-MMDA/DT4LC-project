import { useState } from 'react';
import { Download, Loader2, CheckCircle, X, AlertCircle } from 'lucide-react';
import { useExportLayer } from '../../api/hooks/useGEEData';
import { useAppStore } from '../../store/useAppStore';

interface Layer {
  id: string;
  name: string;
  type: string;
  visible: boolean;
  opacity: number;
  url?: string;
}

interface ExportLayerDialogProps {
  layerId: string;
  layerName: string;
  onClose: () => void;
  allLayers?: Layer[];
}

export function ExportLayerDialog({ layerId, layerName, onClose, allLayers }: ExportLayerDialogProps) {
  const [scale, setScale] = useState(10);
  const [source, setSource] = useState<'auto' | 'gee' | 'microsoft'>('auto');
  const [exportName, setExportName] = useState('');
  const [exportingIndex, setExportingIndex] = useState(0);
  const [completedExports, setCompletedExports] = useState<string[]>([]);
  const [failedExports, setFailedExports] = useState<string[]>([]);
  const exportMutation = useExportLayer();
  const { addAttachment } = useAppStore();

  const isExportAll = layerId === 'all' && allLayers && allLayers.length > 0;
  const totalLayers = isExportAll ? allLayers.length : 1;

  const handleExport = async () => {
    if (isExportAll && allLayers) {
      // Export all layers sequentially
      setCompletedExports([]);
      setFailedExports([]);

      for (let i = 0; i < allLayers.length; i++) {
        setExportingIndex(i);
        try {
          const result = await exportMutation.mutateAsync({
            layer_id: allLayers[i].id,
            scale: scale,
            format: 'geotiff',
            source: source,
            name: exportName.trim() || undefined,
          });

          if (result.ok && result.attachment) {
            addAttachment(result.attachment);
            setCompletedExports((prev) => [...prev, allLayers[i].name]);
          } else {
            setFailedExports((prev) => [...prev, allLayers[i].name]);
          }
        } catch (error) {
          console.error(`Export failed for ${allLayers[i].name}:`, error);
          setFailedExports((prev) => [...prev, allLayers[i].name]);
        }
      }
    } else {
      // Export single layer
      try {
        const result = await exportMutation.mutateAsync({
          layer_id: layerId,
          scale: scale,
          format: 'geotiff',
          source: source,
          name: exportName.trim() || undefined,
        });

        if (result.ok && result.attachment) {
          addAttachment(result.attachment);
        }
      } catch (error) {
        console.error('Export failed:', error);
      }
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999]" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-900 rounded-lg p-6 max-w-md w-full mx-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Export for AI Analysis</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Description */}
        <div className="mb-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Export <span className="font-medium text-gray-900 dark:text-white">"{layerName}"</span> as GeoTIFF for AI analysis.
          </p>

          {/* Export name input */}
          <div className="space-y-2 mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Export Name
            </label>
            <input
              type="text"
              value={exportName}
              onChange={(e) => setExportName(e.target.value)}
              placeholder="e.g., forest_area_study"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={exportMutation.isPending || exportMutation.isSuccess}
            />
          </div>

          {/* Data source selector */}
          <div className="space-y-2 mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Data Source
            </label>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value as 'auto' | 'gee' | 'microsoft')}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={exportMutation.isPending || exportMutation.isSuccess}
            >
              <option value="auto">Auto</option>
              <option value="gee">Google Earth Engine</option>
              <option value="microsoft">Microsoft Planetary Computer</option>
            </select>
          </div>

          {/* Resolution selector */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Resolution
            </label>
            <select
              value={scale}
              onChange={(e) => setScale(Number(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={exportMutation.isPending || exportMutation.isSuccess}
            >
              <option value={10}>10m</option>
              <option value={30}>30m</option>
              <option value={100}>100m</option>
              <option value={250}>250m</option>
            </select>
          </div>
        </div>

        {/* Status messages */}
        <div className="mb-4 min-h-[60px]">
          {exportMutation.isIdle && (
            <div className="flex items-start gap-2 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <Download className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-blue-800 dark:text-blue-200">
                <p className="font-medium">Ready to export</p>
              </div>
            </div>
          )}

          {exportMutation.isPending && (
            <div className="flex items-center gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <Loader2 className="w-5 h-5 text-blue-600 dark:text-blue-400 animate-spin flex-shrink-0" />
              <div className="text-sm text-blue-800 dark:text-blue-200">
                <p className="font-medium">
                  {isExportAll
                    ? `Exporting ${exportingIndex + 1} of ${totalLayers}...`
                    : 'Exporting...'}
                </p>
              </div>
            </div>
          )}

          {exportMutation.isSuccess && exportMutation.data && exportMutation.data.ok && (
            <div className="flex items-start gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-green-800 dark:text-green-200">
                <p className="font-medium mb-1">Export completed!</p>
                {isExportAll ? (
                  <p className="text-xs text-green-700 dark:text-green-300">
                    Exported {completedExports.length} of {totalLayers} layers
                    {failedExports.length > 0 && <><br />{failedExports.length} failed</>}
                  </p>
                ) : (
                  <p className="text-xs text-green-700 dark:text-green-300">
                    {exportMutation.data.size_mb?.toFixed(2)} MB - Added to chat
                  </p>
                )}
              </div>
            </div>
          )}

          {exportMutation.isSuccess && exportMutation.data && !exportMutation.data.ok && (
            <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-red-800 dark:text-red-200">
                <p className="font-medium mb-1">Export failed</p>
                <p className="text-xs text-red-700 dark:text-red-300">
                  {exportMutation.data.error || 'Unknown error occurred'}
                </p>
              </div>
            </div>
          )}

          {exportMutation.isError && (
            <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-red-800 dark:text-red-200">
                <p className="font-medium mb-1">Export failed</p>
                <p className="text-xs text-red-700 dark:text-red-300">
                  {exportMutation.error?.message || 'Unknown error occurred'}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors font-medium"
          >
            {exportMutation.isSuccess && exportMutation.data?.ok ? 'Done' : 'Cancel'}
          </button>
          <button
            onClick={handleExport}
            disabled={exportMutation.isPending || (exportMutation.isSuccess && exportMutation.data?.ok)}
            className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg transition-colors font-medium flex items-center justify-center gap-2"
          >
            {exportMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Exporting...
              </>
            ) : exportMutation.isSuccess && exportMutation.data?.ok ? (
              <>
                <CheckCircle className="w-4 h-4" />
                Exported
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                Export
              </>
            )}
          </button>
        </div>

      </div>
    </div>
  );
}
