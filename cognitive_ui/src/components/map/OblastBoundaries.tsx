import { useEffect } from 'react';
import { Layer, Source, useMap } from 'react-map-gl/maplibre';

const LINE_ID = 'oblast-boundaries-line';

/** Ukrainian oblast boundaries (static GeoJSON in /public), always drawn above imagery. */
export function OblastBoundaries() {
  const { current: mapRef } = useMap();

  useEffect(() => {
    const map = mapRef?.getMap();
    if (!map) return;
    // Raster layers (GEE, GeoTIFF) are added later and would cover the lines; keep the lines on top.
    const keepOnTop = () => {
      const layers = map.getStyle()?.layers;
      if (!layers || !map.getLayer(LINE_ID)) return;
      if (layers[layers.length - 1].id !== LINE_ID) map.moveLayer(LINE_ID);
    };
    map.on('styledata', keepOnTop);
    keepOnTop();
    return () => {
      map.off('styledata', keepOnTop);
    };
  }, [mapRef]);

  return (
    <Source id="oblast-boundaries" type="geojson" data="/ukraine_oblasts.json">
      <Layer
        id={LINE_ID}
        type="line"
        paint={{
          'line-color': '#facc15',
          'line-width': ['interpolate', ['linear'], ['zoom'], 4, 0.8, 10, 2],
          'line-opacity': 0.9,
        }}
      />
    </Source>
  );
}
