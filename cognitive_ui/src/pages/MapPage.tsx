import { useEffect, useMemo } from 'react';
import { MapContainer } from '../components/map/MapContainer';
import { ImageryPanel } from '../components/map/mode/ImageryPanel';
import { sampleLayerOf } from '../components/map/mode/layers';
import { useMapSample } from '../components/map/mode/useMapSample';
import { useLayout } from '../hooks/useMediaQuery';
import { useAppStore } from '../store/useAppStore';

/**
 * /app/map, map mode: the imagery panel beside the
 * map, no panel over another (#67 bug 4). The sample shows under the results in
 * true colour, with its footprint. On a phone the panel is a sheet under the map,
 * collapsed to its header until opened; on a tablet it is narrower.
 */
export function MapPage() {
  const sample = useMapSample();
  const sampleLayer = useMemo(() => sampleLayerOf(sample), [sample]);
  const addBaseLayer = useAppStore((s) => s.addBaseLayer);

  useEffect(() => {
    if (sampleLayer) addBaseLayer(sampleLayer);
  }, [sampleLayer, addBaseLayer]);

  const map = (
    <main aria-label="Map" className="flex-1 min-w-0 min-h-0 relative">
      <MapContainer />
    </main>
  );
  if (useLayout() === 'phone') {
    return (
      <div className="flex-1 min-w-0 flex flex-col">
        {map}
        <ImageryPanel sheet />
      </div>
    );
  }
  return (
    <>
      <ImageryPanel />
      {map}
    </>
  );
}
