import { useState } from 'react';
import { Satellite, Loader2, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { useFetchSentinel2 } from '../../api/hooks/useGEEData';

type DataType = 'rgb' | 'ndvi' | 'ndwi' | 'ndsi';

export function DataFetchPanel() {
  const selectedRegion = useAppStore((state) => state.selectedRegion);
  const addLayer = useAppStore((state) => state.addLayer);
  const toggleDatasetPanelVersion = useAppStore((state) => state.toggleDatasetPanelVersion);

  // Form state
  const [startDate, setStartDate] = useState(() => {
    // Default to 30 days ago
    const date = new Date();
    date.setDate(date.getDate() - 30);
    return date.toISOString().split('T')[0];
  });
  const [endDate, setEndDate] = useState(() => {
    // Default to today
    return new Date().toISOString().split('T')[0];
  });
  const [dataType, setDataType] = useState<DataType>('rgb');
  const [cloudCover, setCloudCover] = useState(20);

  const fetchSentinel2 = useFetchSentinel2();

  const handleFetch = async () => {
    if (!selectedRegion) return;

    try {
      const result = await fetchSentinel2.mutateAsync({
        bbox: selectedRegion.bbox,
        start_date: startDate,
        end_date: endDate,
        data_type: dataType,
        cloud_cover_max: cloudCover,
      });

      if (result.ok && result.tile_url) {
        // Add GEE layer to map
        addLayer({
          id: `gee-${dataType}-${Date.now()}`,
          name: `Sentinel-2 ${dataType.toUpperCase()} (${startDate} to ${endDate})`,
          type: 'gee-tiles',
          visible: true,
          opacity: 1.0,
          url: result.tile_url,
        });
      }
    } catch (error) {
      console.error('Failed to fetch Sentinel-2 data:', error);
    }
  };

  return (
    <div className="absolute bottom-4 left-4 bg-white dark:bg-gray-900 rounded-lg shadow-lg p-4 w-80 z-10">
      <div className="flex items-center gap-2 mb-4">
        <Satellite className="w-5 h-5 text-blue-500" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Fetch Satellite Data
        </h3>
      </div>

      {/* Migration Banner */}
      <div className="flex items-start gap-2 p-3 mb-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <Info className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-sm text-blue-800 dark:text-blue-200 mb-2">
            Try our new Dataset Selection panel with multi-dataset support!
          </p>
          <button
            onClick={toggleDatasetPanelVersion}
            className="text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline"
          >
            Switch to New Panel →
          </button>
        </div>
      </div>

      {!selectedRegion && (
        <div className="flex items-start gap-2 p-3 mb-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <AlertCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-500 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            Please draw or select a region on the map first
          </p>
        </div>
      )}

      <div className="space-y-4">
        {/* Data Type Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Data Type
          </label>
          <div className="grid grid-cols-2 gap-2">
            {(['rgb', 'ndvi', 'ndwi', 'ndsi'] as DataType[]).map((type) => (
              <button
                key={type}
                onClick={() => setDataType(type)}
                className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                  dataType === type
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                {type.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Date Range */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Date Range
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                Start
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                End
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              />
            </div>
          </div>
        </div>

        {/* Cloud Cover Slider */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Max Cloud Cover: {cloudCover}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={cloudCover}
            onChange={(e) => setCloudCover(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Fetch Button */}
        <button
          onClick={handleFetch}
          disabled={!selectedRegion || fetchSentinel2.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
        >
          {fetchSentinel2.isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Fetching...
            </>
          ) : (
            <>
              <Satellite className="w-4 h-4" />
              Fetch Sentinel-2 Data
            </>
          )}
        </button>

        {/* Status Messages */}
        {fetchSentinel2.isSuccess && (
          <div className="flex items-start gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-green-800 dark:text-green-200">
              <p className="font-medium">Data fetched successfully!</p>
              <p className="text-xs mt-1">
                Found {fetchSentinel2.data?.image_count} images. Layer added to map.
              </p>
            </div>
          </div>
        )}

        {fetchSentinel2.isError && (
          <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-red-800 dark:text-red-200">
              <p className="font-medium">Failed to fetch data</p>
              <p className="text-xs mt-1">
                {fetchSentinel2.error?.message || 'Unknown error occurred'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
