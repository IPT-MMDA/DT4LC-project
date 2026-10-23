import { Layer, Source } from 'react-map-gl/maplibre';
import type { RasterLayerSpecification } from 'maplibre-gl';

interface GEETileLayerProps {
  id: string;
  url: string;
  opacity?: number;
  visible?: boolean;
  beforeId?: string;
}

export function GEETileLayer({ id, url, opacity = 1, visible = true, beforeId }: GEETileLayerProps) {
  if (!visible) return null;

  const layerStyle: RasterLayerSpecification = {
    id,
    source: `${id}-source`,
    type: 'raster',
    paint: {
      'raster-opacity': opacity,
      'raster-fade-duration': 0
    }
  };

  return (
    <Source
      id={`${id}-source`}
      type="raster"
      tiles={[url]}
      tileSize={256}
    >
      <Layer {...layerStyle} beforeId={beforeId} />
    </Source>
  );
}
