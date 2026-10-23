import { useState } from 'react';
import { Satellite, ChevronLeft, ChevronRight, Loader2, AlertCircle, MapPin, Trash2, Eraser, ChevronDown, ChevronUp } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { useBulkFetchDatasets, usePersistLayer } from '../../api/hooks/useGEEData';
import { DataSourceSelector } from './DataSourceSelector';
import { BandChannelSelector } from './BandChannelSelector';
import { SpectralIndexSelector } from './SpectralIndexSelector';
import { TimePeriodSelector } from './TimePeriodSelector';
import { QualityFilterControls } from './QualityFilterControls';
import { CustomDatasetUpload } from './CustomDatasetUpload';

interface QuickLocation {
  name: string;
  description: string;
  coordinates: [number, number];
  zoom?: number;
}

function QuickLocationButton({ name, description, coordinates }: QuickLocation) {
  const { mapInstance } = useAppStore();

  const handleClick = () => {
    if (mapInstance) {
      mapInstance.flyTo({
        center: coordinates,
        zoom: 11,
        duration: 2000
      });
    }
  };

  return (
    <button
      onClick={handleClick}
      className="w-full text-left px-3 py-2 text-xs rounded hover:bg-surface2 transition-colors"
      title={description}
    >
      <div className="font-medium text-text">{name}</div>
      <div className="text-xs text-muted mt-0.5">{description}</div>
    </button>
  );
}

export function DatasetSelectionPanel() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showQuickLocations, setShowQuickLocations] = useState(true);
  const [showDrawnRegions, setShowDrawnRegions] = useState(true);

  const {
    selectedRegion,
    selectedDatasetId,
    selectedBands,
    selectedIndices,
    prePeriod,
    postPeriod,
    useNowForPost,
    cloudCoverMax,
    isFetchingDatasets,
    fetchError,
    drawnRegions,
    addLayer,
    setIsFetchingDatasets,
    setFetchError,
    setSelectedDataset,
    setSelectedRegion,
    deleteRegion,
    clearRegions,
  } = useAppStore();

  const bulkFetch = useBulkFetchDatasets();
  const persistLayer = usePersistLayer();

  const calculateFetchCount = () => {
    const bandCount = selectedBands.length > 0 ? 1 : 0;
    const indexCount = selectedIndices.length;
    const totalPerPeriod = bandCount + indexCount;
    return totalPerPeriod * 2;
  };

  const fetchCount = calculateFetchCount();
  const canFetch = selectedRegion && (selectedBands.length > 0 || selectedIndices.length > 0) && !isFetchingDatasets;

  const handleBulkFetch = async () => {
    if (!canFetch || !selectedRegion) return;

    setIsFetchingDatasets(true);
    setFetchError(null);

    try {
      const actualPostPeriod = useNowForPost ? calculateNowPeriod() : postPeriod;

      console.log('Bulk fetch request:', {
        bbox: selectedRegion.bbox,
        dataset_id: selectedDatasetId,
        bands: selectedBands,
        indices: selectedIndices,
        pre_start: prePeriod.start,
        pre_end: prePeriod.end,
        post_start: actualPostPeriod.start,
        post_end: actualPostPeriod.end,
        cloud_cover_max: cloudCoverMax,
        use_now: useNowForPost,
      });

      const result = await bulkFetch.mutateAsync({
        bbox: selectedRegion.bbox,
        dataset_id: selectedDatasetId,
        bands: selectedBands,
        indices: selectedIndices,
        pre_start: prePeriod.start,
        pre_end: prePeriod.end,
        post_start: actualPostPeriod.start,
        post_end: actualPostPeriod.end,
        cloud_cover_max: cloudCoverMax,
        use_now: useNowForPost,
      });

      console.log('Bulk fetch response:', result);

      if (result.ok && result.layers) {
        // Add layers to map and persist metadata for export
        for (const layer of result.layers) {
          addLayer({
            id: layer.layer_id,
            name: layer.layer_name,
            type: 'gee-tiles',
            visible: true,
            opacity: 1.0,
            url: layer.tile_url,
          });

          // Persist layer metadata for future export
          try {
            await persistLayer.mutateAsync({
              layer_id: layer.layer_id,
              layer_name: layer.layer_name,
              dataset_id: selectedDatasetId,
              bands: layer.bands || [],
              indices: layer.index_type ? [layer.index_type] : [],
              period: layer.period,
              bbox: selectedRegion.bbox,
              start_date: layer.period === 'pre' ? prePeriod.start : actualPostPeriod.start,
              end_date: layer.period === 'pre' ? prePeriod.end : actualPostPeriod.end,
              tile_url: layer.tile_url,
              cloud_cover_max: cloudCoverMax,
            });
          } catch (error) {
            console.warn(`Failed to persist metadata for layer ${layer.layer_id}:`, error);
            // Don't fail the whole operation if metadata persistence fails
          }
        }

        const successMsg = `Successfully imported ${result.layers.length} layer${result.layers.length !== 1 ? 's' : ''}`;
        if (result.errors && result.errors.length > 0) {
          setFetchError(`${successMsg}. Some layers failed: ${result.errors.join('; ')}`);
        } else {
          setFetchError(null);
        }
      } else {
        throw new Error(result.error || 'Failed to fetch datasets');
      }
    } catch (error) {
      console.error('Bulk fetch error:', error);
      setFetchError(error instanceof Error ? error.message : 'Failed to fetch datasets');
    } finally {
      setIsFetchingDatasets(false);
    }
  };

  if (isCollapsed) {
    return (
      <div className="absolute left-4 top-4 z-[1001] flex flex-col gap-2">
        <button
          onClick={() => setIsCollapsed(false)}
          className="bg-surface rounded-lg shadow-lg p-2 hover:bg-surface2"
        >
          <ChevronRight className="w-5 h-5" />
        </button>
        <div className="bg-surface rounded-lg shadow-lg p-2 flex flex-col items-center gap-2">
          <Satellite className="w-6 h-6 text-accent" />
          {drawnRegions.length > 0 && (
            <div className="text-xs text-accent font-medium">{drawnRegions.length}</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="absolute left-4 top-4 w-[360px] max-h-[calc(100vh-32px)] overflow-y-auto z-[1001] bg-surface rounded-lg shadow-lg">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Satellite className="w-5 h-5 text-accent" />
            <h3 className="text-lg font-semibold text-text">Map Control</h3>
          </div>
          <button onClick={() => setIsCollapsed(true)} className="p-1 hover:bg-surface2 rounded">
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>

        {/* Quick Locations Section */}
        <div className="mb-4 border border-border rounded-lg">
          <button
            onClick={() => setShowQuickLocations(!showQuickLocations)}
            className="w-full flex items-center justify-between p-3 hover:bg-surface2 rounded-t-lg"
          >
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-muted" />
              <span className="text-sm font-semibold text-text">Quick Locations</span>
            </div>
            {showQuickLocations ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showQuickLocations && (
            <div className="p-3 pt-0 space-y-1">
              <QuickLocationButton
                name="Ukraine - Kahovka"
                description="Kahovka Dam breach"
                coordinates={[33.4, 46.8]}
              />
              <QuickLocationButton
                name="Switzerland - Brienz"
                description="Brienz landslide area"
                coordinates={[8.03, 46.75]}
              />
            </div>
          )}
        </div>

        {/* Drawn Regions Section */}
        <div className="mb-4 border border-border rounded-lg">
          <button
            onClick={() => setShowDrawnRegions(!showDrawnRegions)}
            className="w-full flex items-center justify-between p-3 hover:bg-surface2 rounded-t-lg"
          >
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-accent" />
              <span className="text-sm font-semibold text-text">
                Drawn Regions {drawnRegions.length > 0 && `(${drawnRegions.length})`}
              </span>
            </div>
            {showDrawnRegions ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showDrawnRegions && (
            <div className="p-3 pt-0">
              {drawnRegions.length === 0 ? (
                <p className="text-xs text-muted">
                  Draw a polygon on the map to select a region
                </p>
              ) : (
                <div className="space-y-2">
                  <div className="flex justify-end mb-2">
                    <button
                      onClick={() => {
                        if (confirm(`Delete all ${drawnRegions.length} regions?`)) {
                          clearRegions();
                        }
                      }}
                      className="flex items-center gap-1 px-2 py-1 text-xs text-fail hover:bg-fail-tint rounded"
                      title="Clear all regions"
                    >
                      <Eraser className="w-3 h-3" />
                      Clear All
                    </button>
                  </div>
                  {drawnRegions.map((region) => (
                    <div
                      key={region.id}
                      className={`p-2 rounded border cursor-pointer transition-colors ${
                        selectedRegion?.id === region.id
                          ? 'border-accent bg-accent-tint'
                          : 'border-border hover:border-faint'
                      }`}
                      onClick={() => setSelectedRegion(region)}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-text">
                          {region.name}
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteRegion(region.id);
                          }}
                          className="text-fail hover:text-fail-ink"
                          aria-label="Delete region"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                      <div className="mt-1 text-xs text-muted">
                        {new Date(region.createdAt).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="mb-3 pt-3 border-t border-border">
          <h4 className="text-sm font-semibold text-text mb-3">Dataset Selection</h4>
        </div>

        {!selectedRegion && (
          <div className="flex items-start gap-2 p-3 mb-4 bg-canc-tint border border-canc/30 rounded-lg">
            <AlertCircle className="w-4 h-4 text-canc mt-0.5 flex-shrink-0" />
            <p className="text-sm text-canc-ink">Please draw or select a region on the map first</p>
          </div>
        )}

        {fetchError && (
          <div className="flex items-start gap-2 p-3 mb-4 bg-fail-tint border border-fail/30 rounded-lg">
            <AlertCircle className="w-4 h-4 text-fail mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-sm text-fail-ink">{fetchError}</p>
              <button onClick={() => setFetchError(null)} className="text-xs text-fail hover:underline mt-1">
                Dismiss
              </button>
            </div>
          </div>
        )}

        {selectedDatasetId === 'custom' ? (
          <div>
            <button
              onClick={() => setSelectedDataset('sentinel-2')}
              className="mb-3 text-sm text-accent hover:underline flex items-center gap-1"
            >
              <ChevronLeft className="w-4 h-4" />
              Back to Datasets
            </button>
            <CustomDatasetUpload />
          </div>
        ) : (
          <div className="space-y-3">
            <DataSourceSelector />
            <BandChannelSelector />
            <SpectralIndexSelector />
            <TimePeriodSelector />
            <QualityFilterControls />

            <button
              disabled={!canFetch}
              onClick={handleBulkFetch}
              className="w-full mt-4 px-4 py-2 bg-accent hover:bg-accent/90 disabled:bg-border disabled:cursor-not-allowed text-on-accent font-medium rounded-lg flex items-center justify-center gap-2"
            >
              {isFetchingDatasets ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Fetching...
                </>
              ) : (
                <>
                  <Satellite className="w-4 h-4" />
                  Fetch {fetchCount} Dataset{fetchCount !== 1 ? 's' : ''}
                </>
              )}
            </button>

            {canFetch && fetchCount > 0 && (
              <p className="text-xs text-center text-muted mt-2">
                {selectedBands.length > 0 && `${selectedBands.length} bands`}
                {selectedBands.length > 0 && selectedIndices.length > 0 && ' + '}
                {selectedIndices.length > 0 && `${selectedIndices.length} ${selectedIndices.length === 1 ? 'index' : 'indices'}`}
                {' × 2 periods (Pre + Post)'}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function calculateNowPeriod(): { start: string; end: string } {
  const today = new Date();
  const sevenDaysAgo = new Date(today);
  sevenDaysAgo.setDate(today.getDate() - 7);
  return {
    start: sevenDaysAgo.toISOString().split('T')[0],
    end: today.toISOString().split('T')[0],
  };
}
