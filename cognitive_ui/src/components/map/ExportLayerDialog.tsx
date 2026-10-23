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
        className="bg-surface rounded-lg p-6 max-w-md w-full mx-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text">Export for AI Analysis</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-surface2 rounded transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-muted" />
          </button>
        </div>

        {/* Description */}
        <div className="mb-4">
          <p className="text-sm text-muted mb-3">
            Export <span className="font-medium text-text">"{layerName}"</span> as GeoTIFF for AI analysis.
          </p>

          {/* Export name input */}
          <div className="space-y-2 mb-4">
            <label className="block text-sm font-medium text-text2">
              Export Name
            </label>
            <input
              type="text"
              value={exportName}
              onChange={(e) => setExportName(e.target.value)}
              placeholder="e.g., forest_area_study"
              className="w-full px-3 py-2 border border-border rounded-lg bg-surface text-text placeholder-faint focus:ring-2 focus:ring-accent focus:border-transparent"
              disabled={exportMutation.isPending || exportMutation.isSuccess}
            />
          </div>

          {/* Data source selector */}
          <div className="space-y-2 mb-4">
            <label className="block text-sm font-medium text-text2">
              Data Source
            </label>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value as 'auto' | 'gee' | 'microsoft')}
              className="w-full px-3 py-2 border border-border rounded-lg bg-surface text-text focus:ring-2 focus:ring-accent focus:border-transparent"
              disabled={exportMutation.isPending || exportMutation.isSuccess}
            >
              <option value="auto">Auto</option>
              <option value="gee">Google Earth Engine</option>
              <option value="microsoft">Microsoft Planetary Computer</option>
            </select>
          </div>

          {/* Resolution selector */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-text2">
              Resolution
            </label>
            <select
              value={scale}
              onChange={(e) => setScale(Number(e.target.value))}
              className="w-full px-3 py-2 border border-border rounded-lg bg-surface text-text focus:ring-2 focus:ring-accent focus:border-transparent"
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
            <div className="flex items-start gap-2 p-3 bg-accent-tint border border-accent/30 rounded-lg">
              <Download className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
              <div className="text-sm text-accent-ink">
                <p className="font-medium">Ready to export</p>
              </div>
            </div>
          )}

          {exportMutation.isPending && (
            <div className="flex items-center gap-3 p-3 bg-accent-tint border border-accent/30 rounded-lg">
              <Loader2 className="w-5 h-5 text-accent animate-spin flex-shrink-0" />
              <div className="text-sm text-accent-ink">
                <p className="font-medium">
                  {isExportAll
                    ? `Exporting ${exportingIndex + 1} of ${totalLayers}...`
                    : 'Exporting...'}
                </p>
              </div>
            </div>
          )}

          {exportMutation.isSuccess && exportMutation.data && exportMutation.data.ok && (
            <div className="flex items-start gap-2 p-3 bg-done-tint border border-done/30 rounded-lg">
              <CheckCircle className="w-5 h-5 text-done mt-0.5 flex-shrink-0" />
              <div className="text-sm text-done-ink">
                <p className="font-medium mb-1">Export completed!</p>
                {isExportAll ? (
                  <p className="text-xs text-done-ink">
                    Exported {completedExports.length} of {totalLayers} layers
                    {failedExports.length > 0 && <><br />{failedExports.length} failed</>}
                  </p>
                ) : (
                  <p className="text-xs text-done-ink">
                    {exportMutation.data.size_mb?.toFixed(2)} MB - Added to chat
                  </p>
                )}
              </div>
            </div>
          )}

          {exportMutation.isSuccess && exportMutation.data && !exportMutation.data.ok && (
            <div className="flex items-start gap-2 p-3 bg-fail-tint border border-fail/30 rounded-lg">
              <AlertCircle className="w-5 h-5 text-fail mt-0.5 flex-shrink-0" />
              <div className="text-sm text-fail-ink">
                <p className="font-medium mb-1">Export failed</p>
                <p className="text-xs text-fail-ink">
                  {exportMutation.data.error || 'Unknown error occurred'}
                </p>
              </div>
            </div>
          )}

          {exportMutation.isError && (
            <div className="flex items-start gap-2 p-3 bg-fail-tint border border-fail/30 rounded-lg">
              <AlertCircle className="w-5 h-5 text-fail mt-0.5 flex-shrink-0" />
              <div className="text-sm text-fail-ink">
                <p className="font-medium mb-1">Export failed</p>
                <p className="text-xs text-fail-ink">
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
            className="flex-1 px-4 py-2 border border-border text-text2 rounded-lg hover:bg-surface2 transition-colors font-medium"
          >
            {exportMutation.isSuccess && exportMutation.data?.ok ? 'Done' : 'Cancel'}
          </button>
          <button
            onClick={handleExport}
            disabled={exportMutation.isPending || (exportMutation.isSuccess && exportMutation.data?.ok)}
            className="flex-1 px-4 py-2 bg-accent hover:bg-accent/90 disabled:bg-faint disabled:cursor-not-allowed text-on-accent rounded-lg transition-colors font-medium flex items-center justify-center gap-2"
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
