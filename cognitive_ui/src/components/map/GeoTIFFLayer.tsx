import { useEffect, useState } from 'react';
import { Layer, Source } from 'react-map-gl/maplibre';
import type { RasterLayerSpecification } from 'maplibre-gl';

interface GeoTIFFLayerProps {
  id: string;
  url: string;
  opacity?: number;
  visible?: boolean;
}

export function GeoTIFFLayer({ id, url, opacity = 1, visible = true }: GeoTIFFLayerProps) {
  const [tileUrl, setTileUrl] = useState<string | null>(null);

  useEffect(() => {
    // Backend endpoint to serve GeoTIFF as tiles
    // Format: /v1/tiles/{z}/{x}/{y}?path=<geotiff_path>
    const encodedPath = encodeURIComponent(url);
    const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    setTileUrl(`${baseUrl}/v1/tiles/{z}/{x}/{y}?path=${encodedPath}`);
  }, [url]);

  if (!tileUrl || !visible) return null;

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
      <Layer {...layerStyle} />
    </Source>
  );
}
