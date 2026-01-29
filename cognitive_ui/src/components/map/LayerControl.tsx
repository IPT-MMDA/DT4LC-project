import { useState } from 'react';
import { Eye, EyeOff, Layers, Download, Trash2 } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { ExportLayerDialog } from './ExportLayerDialog';

export function LayerControl() {
  const [exportDialogLayerId, setExportDialogLayerId] = useState<string | null>(null);
  const [exportDialogLayerName, setExportDialogLayerName] = useState<string>('');
  const [exportAllDialogOpen, setExportAllDialogOpen] = useState(false);

  const mapLayers = useAppStore((state) => state.mapLayers);
  const toggleLayerVisibility = useAppStore((state) => state.toggleLayerVisibility);
  const setLayerOpacity = useAppStore((state) => state.setLayerOpacity);
  const removeLayer = useAppStore((state) => state.removeLayer);

  const handleExportClick = (layerId: string, layerName: string) => {
    setExportDialogLayerId(layerId);
    setExportDialogLayerName(layerName);
  };

  const handleCloseDialog = () => {
    setExportDialogLayerId(null);
    setExportDialogLayerName('');
  };

  if (mapLayers.length === 0) {
    return null;
  }

  const geeLayers = mapLayers.filter((layer) => layer.type === 'gee-tiles');

  return (
    <div className="absolute top-4 right-4 bg-white dark:bg-gray-900 rounded-lg shadow-lg p-4 w-64 z-10">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-gray-700 dark:text-gray-300" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Map Layers
          </h3>
        </div>
        {/* Export All button - only show if there are GEE layers */}
        {geeLayers.length > 0 && (
          <button
            onClick={() => setExportAllDialogOpen(true)}
            className="px-2 py-1 text-xs font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors flex items-center gap-1"
            title="Export all layers"
          >
            <Download className="w-3 h-3" />
            Export All
          </button>
        )}
      </div>

      <div className="space-y-3">
        {mapLayers.map((layer) => (
          <div key={layer.id} className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium text-gray-900 dark:text-white truncate flex-1">
                {layer.name}
              </span>
              <div className="flex items-center gap-1">
                {/* Export button for GEE layers */}
                {layer.type === 'gee-tiles' && (
                  <button
                    onClick={() => handleExportClick(layer.id, layer.name)}
                    className="p-1 text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors"
                    title="Export for AI Analysis"
                    aria-label="Export layer for analysis"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                )}
                <button
                  onClick={() => toggleLayerVisibility(layer.id)}
                  className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded transition-colors"
                  aria-label={layer.visible ? 'Hide layer' : 'Show layer'}
                >
                  {layer.visible ? (
                    <Eye className="w-4 h-4" />
                  ) : (
                    <EyeOff className="w-4 h-4" />
                  )}
                </button>
                {/* Delete button */}
                <button
                  onClick={() => removeLayer(layer.id)}
                  className="p-1 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                  aria-label="Delete layer"
                  title="Delete layer"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
            {layer.visible && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
                  Opacity
                </span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={layer.opacity * 100}
                  onChange={(e) => setLayerOpacity(layer.id, parseInt(e.target.value) / 100)}
                  className="w-full h-1 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  aria-label={`Opacity for ${layer.name}`}
                />
                <span className="text-xs text-gray-500 dark:text-gray-400 w-8 text-right">
                  {Math.round(layer.opacity * 100)}%
                </span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Export dialog for single layer */}
      {exportDialogLayerId && (
        <ExportLayerDialog
          layerId={exportDialogLayerId}
          layerName={exportDialogLayerName}
          onClose={handleCloseDialog}
        />
      )}

      {/* Export All dialog */}
      {exportAllDialogOpen && (
        <ExportLayerDialog
          layerId="all"
          layerName="All Layers"
          onClose={() => setExportAllDialogOpen(false)}
          allLayers={geeLayers}
        />
      )}
    </div>
  );
}
