import { Layer, Source } from 'react-map-gl/maplibre';
import { useAppStore } from '../../store/useAppStore';
import { GeoTIFFLayer } from './GeoTIFFLayer';
import { GEETileLayer } from './GEETileLayer';
import { useMapColours } from './mapColours';

/** An invisible layer the rasters stack under, so drawn regions and footprints stay on top. */
export const RASTER_ANCHOR = 'dt4lc-raster-anchor';

export function RasterAnchor() {
  return <Layer id={RASTER_ANCHOR} type="background" paint={{ 'background-opacity': 0 }} />;
}

interface JobLayersManagerProps {
  /** Compare: this raster only (the left one on the map, the right one on the second map). */
  only?: string;
  /** The rasters' outlines; Compare's second map leaves them to the first. */
  footprints?: boolean;
}

/**
 * The store's raster layers in `mapLayers` order (last on top). MapLibre puts a new
 * layer on top, so each is placed under the one above it (`beforeId`), the top one
 * under RASTER_ANCHOR, which the map renders first.
 */
export function JobLayersManager({ only, footprints = true }: JobLayersManagerProps = {}) {
  const mapLayers = useAppStore((state) => state.mapLayers);
  const colours = useMapColours();
  const visible = mapLayers.filter((l) => l.url && l.visible);
  const topFirst = visible.filter((l) => !only || l.id === only).reverse();

  return (
    <>
      {topFirst.map((layer, i) => {
        const beforeId = i === 0 ? RASTER_ANCHOR : topFirst[i - 1].id;
        const props = { id: layer.id, url: layer.url as string, opacity: layer.opacity, beforeId };
        if (layer.type === 'gee-tiles') return <GEETileLayer key={layer.id} {...props} />;
        if (layer.type === 'raster') return <GeoTIFFLayer key={layer.id} {...props} />;
        return null;
      })}
      {footprints &&
        visible
          .filter((l) => l.footprint && l.footprint.length > 2)
          .map((layer) => (
            <Source
              key={`${layer.id}-footprint`}
              id={`${layer.id}-footprint`}
              type="geojson"
              data={{
                type: 'Feature',
                properties: {},
                geometry: { type: 'Polygon', coordinates: [[...(layer.footprint ?? []), (layer.footprint ?? [])[0]]] },
              }}
            >
              <Layer
                id={`${layer.id}-footprint-line`}
                type="line"
                paint={{ 'line-color': colours.footprint, 'line-width': 1.5, 'line-dasharray': [3, 2] }}
              />
            </Source>
          ))}
    </>
  );
}
