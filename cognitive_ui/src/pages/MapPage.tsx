import { useNavigate } from 'react-router-dom';
import { MessageSquare } from 'lucide-react';
import { MapContainer } from '../components/map/MapContainer';
import { RegionPanel } from '../components/map/RegionPanel';
import { LayerControl } from '../components/map/LayerControl';
import { DataFetchPanel } from '../components/map/DataFetchPanel';
import { DatasetSelectionPanel } from '../components/map/DatasetSelectionPanel';
import { useAppStore } from '../store/useAppStore';

export function MapPage() {
  const navigate = useNavigate();
  const useNewDatasetPanel = useAppStore((state) => state.useNewDatasetPanel);
  const drawnRegions = useAppStore((state) => state.drawnRegions);
  const mapLayers = useAppStore((state) => state.mapLayers);
  const selectedRegion = useAppStore((state) => state.selectedRegion);
  const setPendingAnalysis = useAppStore((state) => state.setPendingAnalysis);

  const hasData = drawnRegions.length > 0 || mapLayers.length > 0;

  const handleSendToChat = () => {
    // Navigate to Chat with bbox pre-populated if available
    const region = selectedRegion || (drawnRegions.length > 0 ? drawnRegions[0] : null);
    
    if (region) {
      setPendingAnalysis({
        bbox: region.bbox,
      });
    } else {
      setPendingAnalysis({
        message: 'Analyze the loaded map layers',
      });
    }
    navigate('/chat');
  };

  return (
    <div className="h-full w-full relative">
      <MapContainer />
      {!useNewDatasetPanel && <RegionPanel />}
      <LayerControl />
      {useNewDatasetPanel ? <DatasetSelectionPanel /> : <DataFetchPanel />}

      {hasData && (
        <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-10">
          <button
            onClick={handleSendToChat}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg hover:shadow-xl transition-all font-medium"
          >
            <MessageSquare className="w-5 h-5" />
            Send to Chat
          </button>
        </div>
      )}
    </div>
  );
}
