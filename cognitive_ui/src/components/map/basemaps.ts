import type { StyleSpecification } from 'maplibre-gl';

// Free basemaps that need no API key (moved from MapContainer for the Basemap ▾ picker
// and Compare's second map).

const SATELLITE: StyleSpecification = {
  version: 8,
  sources: {
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: 'Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community'
    }
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-satellite',
      minzoom: 0,
      maxzoom: 22
    }
  ]
};

export const BASEMAPS: { id: string; label: string; style: string | StyleSpecification }[] = [
  { id: 'satellite', label: 'Satellite', style: SATELLITE },
  { id: 'streets', label: 'Streets', style: 'https://demotiles.maplibre.org/style.json' },
  { id: 'terrain', label: 'Terrain', style: 'https://demotiles.maplibre.org/style.json' },
];

export function basemapStyle(id: string): string | StyleSpecification {
  return (BASEMAPS.find((b) => b.id === id) ?? BASEMAPS[1]).style;
}
