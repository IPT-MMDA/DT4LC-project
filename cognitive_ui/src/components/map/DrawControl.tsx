import { useRef, useImperativeHandle, forwardRef } from 'react';
import { useControl } from 'react-map-gl/maplibre';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';

interface DrawControlProps {
  onCreate?: (features: any) => void;
  onUpdate?: (features: any) => void;
  onDelete?: (features: any) => void;
}

export interface DrawControlRef {
  changeMode: (mode: string) => void;
  deleteAll: () => void;
  getSelected: () => any;
}

export const DrawControl = forwardRef<DrawControlRef, DrawControlProps>(
  ({ onCreate, onUpdate, onDelete }, ref) => {
    const drawRef = useRef<MapboxDraw | null>(null);

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
          styles: [
            // Polygon fill while drawing
            {
              id: 'gl-draw-polygon-fill-inactive',
              type: 'fill',
              filter: ['all', ['==', 'active', 'false'], ['==', '$type', 'Polygon']],
              paint: {
                'fill-color': '#3b82f6',
                'fill-outline-color': '#3b82f6',
                'fill-opacity': 0.1
              }
            },
            {
              id: 'gl-draw-polygon-fill-active',
              type: 'fill',
              filter: ['all', ['==', 'active', 'true'], ['==', '$type', 'Polygon']],
              paint: {
                'fill-color': '#fbbf24',
                'fill-outline-color': '#fbbf24',
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
                'line-color': '#3b82f6',
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
                'line-color': '#fbbf24',
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
                'circle-color': '#fff',
                'circle-stroke-color': '#3b82f6',
                'circle-stroke-width': 2
              }
            },
            {
              id: 'gl-draw-polygon-and-line-vertex-active',
              type: 'circle',
              filter: ['all', ['==', 'meta', 'vertex'], ['==', '$type', 'Point'], ['==', 'active', 'true']],
              paint: {
                'circle-radius': 6,
                'circle-color': '#fff',
                'circle-stroke-color': '#fbbf24',
                'circle-stroke-width': 3
              }
            }
          ]
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
      },
      getSelected: () => {
        if (drawRef.current) {
          return (drawRef.current as any).getSelected();
        }
        return null;
      }
    }));

    return null;
  }
);

DrawControl.displayName = 'DrawControl';
