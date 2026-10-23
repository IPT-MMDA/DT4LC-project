import { ChevronDown } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { DATASETS, getDefaultBands } from '../../config/datasets';

export function DataSourceSelector() {
  const { selectedDatasetId, setSelectedDataset, setBands } = useAppStore();

  const handleDatasetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newDatasetId = e.target.value;
    setSelectedDataset(newDatasetId);
    const defaultBands = getDefaultBands(newDatasetId);
    setBands(defaultBands);
  };

  const selectedDataset = DATASETS[selectedDatasetId];

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-text2">
        Data Source
      </label>

      <div className="relative">
        <select
          value={selectedDatasetId}
          onChange={handleDatasetChange}
          className="w-full px-3 py-2 text-sm border border-border rounded-lg bg-surface text-text appearance-none cursor-pointer focus:ring-2 focus:ring-accent focus:border-accent"
        >
          {Object.values(DATASETS).map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name} ({dataset.spatialResolution})
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-faint pointer-events-none" />
      </div>

      {selectedDataset && (
        <div className="p-3 bg-surface2 rounded-lg text-xs text-muted">
          <div>{selectedDataset.description}</div>
          <div className="mt-1">Temporal: {selectedDataset.temporalResolution}</div>
        </div>
      )}
    </div>
  );
}
