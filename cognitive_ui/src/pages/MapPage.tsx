import { MapContainer } from '../components/map/MapContainer';
import { RegionPanel } from '../components/map/RegionPanel';
import { LayerControl } from '../components/map/LayerControl';
import { DataFetchPanel } from '../components/map/DataFetchPanel';
import { DatasetSelectionPanel } from '../components/map/DatasetSelectionPanel';
import { useAppStore } from '../store/useAppStore';

export function MapPage() {
  const useNewDatasetPanel = useAppStore((state) => state.useNewDatasetPanel);

  return (
    <div className="h-full w-full relative">
      <MapContainer />
      {!useNewDatasetPanel && <RegionPanel />}
      <LayerControl />
      {useNewDatasetPanel ? <DatasetSelectionPanel /> : <DataFetchPanel />}
    </div>
  );
}
