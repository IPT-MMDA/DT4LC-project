import { useState } from 'react';
import { useBulkFetchDatasets, usePersistLayer } from '../../../api/hooks/useGEEData';
import { useAppStore, type DrawnRegion, type Imagery } from '../../../store/useAppStore';
import { dateRange } from '../../chat/regionChip';
import { bulkFetchRequest } from './imagery';

/**
 * Fetch to map: the region's imagery from Earth Engine through `POST /v1/gee/bulk-fetch`
 * (before and after composites), each added as a tile layer and its metadata persisted
 * for Export. Every failure ends in `error`, shown in the panel, never silently.
 */
export function useFetchToMap() {
  const bulk = useBulkFetchDatasets();
  const persist = usePersistLayer();
  const addResultLayer = useAppStore((s) => s.addResultLayer);
  const [error, setError] = useState<string | null>(null);

  const run = async (region: DrawnRegion, imagery: Imagery) => {
    setError(null);
    const request = bulkFetchRequest(region, imagery);
    try {
      const result = await bulk.mutateAsync(request);
      if (!result.ok || result.layers.length === 0) {
        throw new Error(result.error || 'Earth Engine returned no imagery for this region and these dates');
      }
      for (const layer of result.layers) {
        const [start, end] = layer.period === 'pre' ? [request.pre_start, request.pre_end] : [request.post_start ?? '', request.post_end ?? ''];
        addResultLayer({
          id: layer.layer_id,
          name: layer.layer_name,
          type: 'gee-tiles',
          visible: true,
          opacity: 1,
          url: layer.tile_url,
          bounds: region.bbox,
          source: 'earth-engine',
          subtitle: `${imagery.dataset} · ${layer.period === 'pre' ? 'before' : 'after'} · ${dateRange(start, end)}`,
        });
        try {
          await persist.mutateAsync({
            layer_id: layer.layer_id,
            layer_name: layer.layer_name,
            dataset_id: request.dataset_id,
            bands: layer.bands ?? [],
            indices: layer.index_type ? [layer.index_type] : [],
            period: layer.period,
            bbox: region.bbox,
            start_date: start,
            end_date: end,
            tile_url: layer.tile_url,
            cloud_cover_max: imagery.cloud_max,
          });
        } catch {
          // Only Export needs the stored metadata; the layer is on the map.
        }
      }
      if (result.errors && result.errors.length > 0) setError(`Some layers failed: ${result.errors.join('; ')}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Earth Engine didn't return imagery");
    }
  };

  return { run, pending: bulk.isPending, error, clearError: () => setError(null) };
}
