import { useEffect, useMemo } from 'react';
import { MapContainer } from '../components/map/MapContainer';
import { ImageryPanel } from '../components/map/mode/ImageryPanel';
import { sampleLayerOf } from '../components/map/mode/layers';
import { useMapSample } from '../components/map/mode/useMapSample';
import { useAppStore } from '../store/useAppStore';

/**
 * /app/map, map mode (design/screens/map-mode.png): the imagery panel beside the
 * map, no panel over another (#67 bug 4). The sample shows under the results in
 * true colour, with its footprint.
 */
export function MapPage() {
  const sample = useMapSample();
  const sampleLayer = useMemo(() => sampleLayerOf(sample), [sample]);
  const addBaseLayer = useAppStore((s) => s.addBaseLayer);

  useEffect(() => {
    if (sampleLayer) addBaseLayer(sampleLayer);
  }, [sampleLayer, addBaseLayer]);

  return (
    <>
      <ImageryPanel />
      <section aria-label="Map" className="flex-1 min-w-0 relative">
        <MapContainer />
      </section>
    </>
  );
}
