import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { useControl, useMap } from 'react-map-gl/maplibre';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import type { IControl } from 'maplibre-gl';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import { useMapColours, type MapColours } from './mapColours';
import { RECTANGLE_MODE, rectangleMode } from './rectangleMode';

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

/** A draw.create / draw.update / draw.delete event: the features concerned. */
export interface DrawFeaturesEvent {
  features: GeoJSON.Feature[];
}

type DrawListener = (event: DrawFeaturesEvent) => void;

interface DrawControlProps {
  onCreate?: DrawListener;
  onUpdate?: DrawListener;
  onDelete?: DrawListener;
}

/** The draw modes the map's toolbar uses: Pan, Polygon, Rectangle. */
export type DrawToolMode = 'simple_select' | 'draw_polygon' | typeof RECTANGLE_MODE;

export interface DrawControlRef {
  changeMode: (mode: DrawToolMode) => void;
  deleteAll: () => void;
}

export const DrawControl = forwardRef<DrawControlRef, DrawControlProps>(
  ({ onCreate, onUpdate, onDelete }, ref) => {
    const drawRef = useRef<MapboxDraw | null>(null);
    const colours = useMapColours();
    const { current: mapRef } = useMap();

    useControl<IControl>(
      () => {
        const instance = new MapboxDraw({
          displayControlsDefault: false,
          controls: {},
          defaultMode: 'simple_select',
          // Configure drawing behavior
          modes: {
            ...MapboxDraw.modes,
            [RECTANGLE_MODE]: rectangleMode,
          },
          // Styling for drawn features
          styles: drawStyles(colours),
        });
        drawRef.current = instance;
        // MapboxDraw is typed against mapbox-gl's IControl; MapLibre's has the same shape.
        return instance as unknown as IControl;
      },
      ({ map }) => {
        if (onCreate) map.on('draw.create', onCreate);
        if (onUpdate) map.on('draw.update', onUpdate);
        if (onDelete) map.on('draw.delete', onDelete);
      },
      ({ map }) => {
        if (onCreate) map.off('draw.create', onCreate);
        if (onUpdate) map.off('draw.update', onUpdate);
        if (onDelete) map.off('draw.delete', onDelete);
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
      changeMode: (mode: DrawToolMode) => {
        const draw = drawRef.current;
        if (!draw) return;
        if (mode === 'simple_select') draw.changeMode('simple_select');
        else if (mode === 'draw_polygon') draw.changeMode('draw_polygon');
        else draw.changeMode(RECTANGLE_MODE);
      },
      deleteAll: () => {
        drawRef.current?.deleteAll();
      }
    }));

    return null;
  }
);

DrawControl.displayName = 'DrawControl';
