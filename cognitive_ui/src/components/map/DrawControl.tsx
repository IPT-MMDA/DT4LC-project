import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { useControl, useMap } from 'react-map-gl/maplibre';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import { useMapColours, type MapColours } from './mapColours';

// Styling for drawn features, from the design tokens (see mapColours.ts).
function drawStyles(c: MapColours) {
  return [
    // Polygon fill while drawing
    {
      id: 'gl-draw-polygon-fill-inactive',
      type: 'fill',
      filter: ['all', ['==', 'active', 'false'], ['==', '$type', 'Polygon']],
      paint: {
        'fill-color': c.regionSelected,
        'fill-outline-color': c.regionSelected,
        'fill-opacity': 0.1
      }
    },
    {
      id: 'gl-draw-polygon-fill-active',
      type: 'fill',
      filter: ['all', ['==', 'active', 'true'], ['==', '$type', 'Polygon']],
      paint: {
        'fill-color': c.drawing,
        'fill-outline-color': c.drawing,
        'fill-opacity': 0.1
      }
    },
    // Polygon outline
    {
      id: 'gl-draw-polygon-stroke-inactive',
      type: 'line',
      filter: ['all', ['==', 'active', 'false'], ['==', '$type', 'Polygon']],
      layout: {
        'line-cap': 'round',
        'line-join': 'round'
      },
      paint: {
        'line-color': c.regionSelected,
        'line-width': 2
      }
    },
    {
      id: 'gl-draw-polygon-stroke-active',
      type: 'line',
      filter: ['all', ['==', 'active', 'true'], ['==', '$type', 'Polygon']],
      layout: {
        'line-cap': 'round',
        'line-join': 'round'
      },
      paint: {
        'line-color': c.drawing,
        'line-width': 2
      }
    },
    // Vertex points
    {
      id: 'gl-draw-polygon-and-line-vertex-inactive',
      type: 'circle',
      filter: ['all', ['==', 'meta', 'vertex'], ['==', '$type', 'Point']],
      paint: {
        'circle-radius': 5,
        'circle-color': c.vertexFill,
        'circle-stroke-color': c.regionSelected,
        'circle-stroke-width': 2
      }
    },
    {
      id: 'gl-draw-polygon-and-line-vertex-active',
      type: 'circle',
      filter: ['all', ['==', 'meta', 'vertex'], ['==', '$type', 'Point'], ['==', 'active', 'true']],
      paint: {
        'circle-radius': 6,
        'circle-color': c.vertexFill,
        'circle-stroke-color': c.drawing,
        'circle-stroke-width': 3
      }
    }
  ];
}

interface DrawControlProps {
  onCreate?: (features: any) => void;
  onUpdate?: (features: any) => void;
  onDelete?: (features: any) => void;
}

export interface DrawControlRef {
  changeMode: (mode: string) => void;
  deleteAll: () => void;
}

export const DrawControl = forwardRef<DrawControlRef, DrawControlProps>(
  ({ onCreate, onUpdate, onDelete }, ref) => {
    const drawRef = useRef<MapboxDraw | null>(null);
    const colours = useMapColours();
    const { current: mapRef } = useMap();

    useControl<MapboxDraw | any>(
      () => {
        const instance = new (MapboxDraw as any)({
          displayControlsDefault: false,
          controls: {},
          defaultMode: 'simple_select',
          // Configure drawing behavior
          modes: {
            ...MapboxDraw.modes,
          },
          // Styling for drawn features
          styles: drawStyles(colours),
        });
        drawRef.current = instance;
        return instance as any;
      },
      ({ map }) => {
        if (onCreate) map.on('draw.create', onCreate as any);
        if (onUpdate) map.on('draw.update', onUpdate as any);
        if (onDelete) map.on('draw.delete', onDelete as any);
      },
      ({ map }) => {
        if (onCreate) map.off('draw.create', onCreate as any);
        if (onUpdate) map.off('draw.update', onUpdate as any);
        if (onDelete) map.off('draw.delete', onDelete as any);
      }
    );

    // MapboxDraw takes its styles once; repaint its layers (`<id>.cold` / `<id>.hot`)
    // when the theme changes.
    useEffect(() => {
      const map = mapRef?.getMap();
      if (!map) return;
      for (const style of drawStyles(colours)) {
        for (const id of [`${style.id}.cold`, `${style.id}.hot`]) {
          if (!map.getLayer(id)) continue;
          for (const [property, value] of Object.entries(style.paint)) {
            map.setPaintProperty(id, property, value);
          }
        }
      }
    }, [mapRef, colours]);

    useImperativeHandle(ref, () => ({
      changeMode: (mode: string) => {
        if (drawRef.current) {
          (drawRef.current as any).changeMode(mode);
        }
      },
      deleteAll: () => {
        if (drawRef.current) {
          (drawRef.current as any).deleteAll();
        }
      }
    }));

    return null;
  }
);

DrawControl.displayName = 'DrawControl';
