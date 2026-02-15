import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { getDatasetConfig } from '../../config/datasets';

export function BandChannelSelector() {
  const [isExpanded, setIsExpanded] = useState(true);
  const { selectedDatasetId, selectedBands, toggleBand, setBands } = useAppStore();

  const dataset = getDatasetConfig(selectedDatasetId);

  if (!dataset || dataset.bands.length === 0) {
    return null;
  }

  const handleSelectAll = () => {
    setBands(dataset.bands.map((b) => b.id));
  };

  const handleClear = () => {
    setBands([]);
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-t-lg"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-900 dark:text-white">Bands/Channels</span>
          {selectedBands.length > 0 && (
            <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-medium rounded">
              {selectedBands.length} selected
            </span>
          )}
        </div>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isExpanded && (
        <div className="p-3 pt-0 space-y-2">
          <div className="flex gap-2 mb-3">
            <button
              onClick={handleSelectAll}
              className="px-3 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              Select All
            </button>
            <button
              onClick={handleClear}
              className="px-3 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              Clear
            </button>
          </div>

          <div className="space-y-2">
            {dataset.bands.map((band) => (
              <label key={band.id} className="flex items-start gap-2 cursor-pointer p-2 hover:bg-gray-50 dark:hover:bg-gray-800 rounded">
                <input
                  type="checkbox"
                  checked={selectedBands.includes(band.id)}
                  onChange={() => toggleBand(band.id)}
                  className="mt-1"
                />
                <div className="flex-1">
                  <div className="text-sm text-gray-900 dark:text-white">
                    {band.id} - {band.name}
                    {band.wavelength && <span className="text-gray-500 dark:text-gray-400 ml-1">({band.wavelength})</span>}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">{band.description}</div>
                </div>
              </label>
            ))}
          </div>

          {selectedBands.length === 0 && (
            <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
              Select at least one band or spectral index to fetch data
            </p>
          )}
        </div>
      )}
    </div>
  );
}
