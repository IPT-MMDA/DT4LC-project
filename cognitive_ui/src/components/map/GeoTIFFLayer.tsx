import { Layer, Source } from 'react-map-gl/maplibre';
import type { RasterLayerSpecification } from 'maplibre-gl';
import { API_BASE_URL } from '../../api/client';

interface GeoTIFFLayerProps {
  id: string;
  url: string;  // the raster's server path
  opacity?: number;
  visible?: boolean;
  beforeId?: string;
}

/** A server raster as tiles: /v1/tiles/{z}/{x}/{y}?path=<path>, styled by the server. */
export function GeoTIFFLayer({ id, url, opacity = 1, visible = true, beforeId }: GeoTIFFLayerProps) {
  if (!visible) return null;

  const tileUrl = `${API_BASE_URL}/v1/tiles/{z}/{x}/{y}?path=${encodeURIComponent(url)}`;
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
      tiles={[tileUrl]}
      tileSize={256}
    >
      <Layer {...layerStyle} beforeId={beforeId} />
    </Source>
  );
}
