import { Trash2, MapPin, Eraser } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export function RegionPanel() {
  const drawnRegions = useAppStore((state) => state.drawnRegions);
  const selectedRegion = useAppStore((state) => state.selectedRegion);
  const setSelectedRegion = useAppStore((state) => state.setSelectedRegion);
  const deleteRegion = useAppStore((state) => state.deleteRegion);
  const clearRegions = useAppStore((state) => state.clearRegions);

  return (
    <div className="absolute top-20 left-4 bg-surface rounded-lg shadow-lg p-4 w-80 max-h-96 overflow-y-auto z-10">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-text">
          Drawn Regions
        </h3>
        {drawnRegions.length > 0 && (
          <button
            onClick={() => {
              if (confirm(`Delete all ${drawnRegions.length} regions?`)) {
                clearRegions();
              }
            }}
            className="flex items-center gap-1 px-2 py-1 text-xs text-fail hover:bg-fail-tint rounded transition-colors"
            title="Clear all regions"
          >
            <Eraser className="w-3 h-3" />
            Clear All
          </button>
        )}
      </div>

      {drawnRegions.length === 0 ? (
        <p className="text-sm text-muted">
          Draw a polygon on the map to select a region
        </p>
      ) : (
        <div className="space-y-2">
          {drawnRegions.map((region) => (
            <div
              key={region.id}
              className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                selectedRegion?.id === region.id
                  ? 'border-accent bg-accent-tint'
                  : 'border-border hover:border-faint'
              }`}
              onClick={() => setSelectedRegion(region)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-accent" />
                  <span className="text-sm font-medium text-text">
                    {region.name}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteRegion(region.id);
                  }}
                  className="text-fail hover:text-fail-ink transition-colors"
                  aria-label="Delete region"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              <div className="mt-1 text-xs text-muted">
                BBox: [{region.bbox.map(v => v.toFixed(4)).join(', ')}]
              </div>
              <div className="mt-1 text-xs text-faint">
                {new Date(region.createdAt).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
