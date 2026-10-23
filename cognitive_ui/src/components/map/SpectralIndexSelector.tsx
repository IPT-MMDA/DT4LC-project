import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { supportsIndex, getDatasetConfig } from '../../config/datasets';

const INDEX_INFO = {
  ndvi: {
    name: 'NDVI',
    fullName: 'Normalized Difference Vegetation Index',
    description: 'Vegetation health and density (higher = healthier)',
  },
  ndwi: {
    name: 'NDWI',
    fullName: 'Normalized Difference Water Index',
    description: 'Water content and open water detection',
  },
  ndsi: {
    name: 'NDSI',
    fullName: 'Normalized Difference Snow Index',
    description: 'Snow and ice cover detection',
  },
};

type IndexType = 'ndvi' | 'ndwi' | 'ndsi';

export function SpectralIndexSelector() {
  const [isExpanded, setIsExpanded] = useState(false);
  const { selectedDatasetId, selectedIndices, toggleIndex } = useAppStore();

  const dataset = getDatasetConfig(selectedDatasetId);

  if (!dataset || dataset.supportedIndices.length === 0) {
    return null;
  }

  return (
    <div className="border border-border rounded-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-surface2 rounded-t-lg"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-text">Spectral Indices (Optional)</span>
          {selectedIndices.length > 0 && (
            <span className="px-2 py-0.5 bg-accent-tint text-accent-ink text-xs font-medium rounded">
              {selectedIndices.length} selected
            </span>
          )}
        </div>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isExpanded && (
        <div className="p-3 pt-0 space-y-2">
          <p className="text-xs text-muted mb-3">
            Each index will be fetched as a separate layer
          </p>

          <div className="space-y-2">
            {(Object.keys(INDEX_INFO) as IndexType[]).map((indexType) => {
              const info = INDEX_INFO[indexType];
              const isSupported = supportsIndex(selectedDatasetId, indexType);

              return (
                <label
                  key={indexType}
                  className={`flex items-start gap-2 p-2 rounded ${
                    isSupported ? 'cursor-pointer hover:bg-surface2' : 'opacity-50 cursor-not-allowed'
                  }`}
                >
                  <input
                    type="checkbox"
                    disabled={!isSupported}
                    checked={selectedIndices.includes(indexType)}
                    onChange={() => toggleIndex(indexType)}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="text-sm text-text">
                      {info.name}
                      {!isSupported && <span className="text-faint ml-1">(Not supported)</span>}
                    </div>
                    <div className="text-xs text-muted">{info.description}</div>
                  </div>
                </label>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
