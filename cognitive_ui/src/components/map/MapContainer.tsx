import { useRef, useState, useEffect } from 'react';
import Map, { MapRef, NavigationControl, ScaleControl, Source, Layer } from 'react-map-gl/maplibre';
import type { LngLatLike } from 'maplibre-gl';
import * as turf from '@turf/turf';
import { useAppStore, type MapTool } from '../../store/useAppStore';
import { basemapStyle } from './basemaps';
import { DrawControl, type DrawControlRef, type DrawFeaturesEvent, type DrawToolMode } from './DrawControl';
import { JobLayersManager, RasterAnchor } from './JobLayersManager';
import { OblastBoundaries } from './OblastBoundaries';
import { useMapColours } from './mapColours';
import { CompareSwipe } from './mode/CompareSwipe';
import { compareCandidates } from './mode/layers';
import { MapControls } from './mode/MapControls';
import { MapLegend } from './mode/MapLegend';
import { MapToolbar } from './mode/MapToolbar';
import { RegionLabel } from './mode/RegionLabel';
import 'maplibre-gl/dist/maplibre-gl.css';

function lngLatOf(c: LngLatLike): { lng: number; lat: number } {
  if (Array.isArray(c)) return { lng: c[0], lat: c[1] };
  return 'lng' in c ? { lng: c.lng, lat: c.lat } : { lng: c.lon, lat: c.lat };
}

const DRAW_MODE: Record<MapTool, DrawToolMode> = { pan: 'simple_select', polygon: 'draw_polygon', rectangle: 'draw_rectangle' };

/**
 * The map. `docked`: the Workspace's map pane at ≥1440 px, without the drawing
 * toolbar and scale bar. Otherwise map mode (/app/map): the toolbar, coordinates,
 * Fit to result, Compare, Basemap ▾, the region label and the legend on the map.
 */
export function MapContainer({ docked = false }: { docked?: boolean }) {
  const mapRef = useRef<MapRef>(null);
  const drawRef = useRef<DrawControlRef>(null);
  const isProcessingDraw = useRef(false);
  const currentBasemap = useAppStore((state) => state.currentBasemap);
  const addRegion = useAppStore((state) => state.addRegion);
  const drawnRegions = useAppStore((state) => state.drawnRegions);
  const selectedRegion = useAppStore((state) => state.selectedRegion);
  const regionOpacity = useAppStore((state) => state.regionOpacity);
  const fitRequest = useAppStore((state) => state.fitRequest);
  const clearFitRequest = useAppStore((state) => state.clearFitRequest);
  const mapTool = useAppStore((state) => state.mapTool);
  const setMapTool = useAppStore((state) => state.setMapTool);
  const compare = useAppStore((state) => state.compare);
  const [loaded, setLoaded] = useState(false);
  const colours = useMapColours();
  const setMapView = useAppStore((state) => state.setMapView);
  const [viewState, setViewState] = useState(() => useAppStore.getState().mapView ?? {
    longitude: 33.4,  // Ukraine - Kahovka region
    latitude: 46.8,
    zoom: 10
  });
  const mapStyle = basemapStyle(currentBasemap);
  const mapLayers = useAppStore((state) => state.mapLayers);
  const setCompare = useAppStore((state) => state.setCompare);
  // Compare applies while both of its layers are on the map.
  const shown = compareCandidates(mapLayers).map((l) => l.id);
  const comparing = !docked && compare !== null && shown.includes(compare.left) && shown.includes(compare.right);

  // Zoom to a result once the map can (On map, Fit to result): the request outlives a
  // navigation to /app/map, whose map consumes it after loading.
  useEffect(() => {
    const map = mapRef.current;
    if (!fitRequest || !loaded || !map) return;
    const [west, south, east, north] = fitRequest;
    map.fitBounds([[west, south], [east, north]], { padding: 48, duration: 800 });
    // The target view, in case this map goes (Open full map) before the animation ends.
    const camera = map.cameraForBounds?.([[west, south], [east, north]], { padding: 48 });
    if (camera?.center && camera.zoom !== undefined) {
      const { lng, lat } = lngLatOf(camera.center);
      setMapView({ longitude: lng, latitude: lat, zoom: camera.zoom });
    }
    clearFitRequest();
  }, [fitRequest, loaded, clearFitRequest, setMapView]);

  // The toolbar's tool (and Draw new) as the draw control's mode, in map mode only.
  useEffect(() => {
    if (docked || !loaded) return;
    drawRef.current?.changeMode(DRAW_MODE[mapTool]);
  }, [docked, loaded, mapTool]);

  // Leaving map mode ends any drawing and Compare.
  useEffect(() => () => {
    if (docked) return;
    setMapTool('pan');
    setCompare(null);
  }, [docked, setMapTool, setCompare]);

  // Handle Escape key to cancel drawing
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && mapTool !== 'pan') setMapTool('pan');
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mapTool, setMapTool]);

  const handleDrawCreate = (event: DrawFeaturesEvent) => {
    // Prevent duplicate processing if event fires multiple times
    if (isProcessingDraw.current) {
      console.log('Ignoring duplicate draw.create event');
      return;
    }

    isProcessingDraw.current = true;

    try {
      const features = event.features;
      if (features.length === 0) return;

      const feature = features[0];
      const bbox = turf.bbox(feature);

      const region = {
        id: `region-${Date.now()}`,
        name: `Region ${new Date().toLocaleTimeString()}`,
        type: (feature.geometry.type === 'Polygon' ? 'polygon' : 'bbox') as 'bbox' | 'polygon',
        geometry: feature,
        bbox: bbox as [number, number, number, number],
        createdAt: new Date().toISOString()
      };

      addRegion(region);

      // Clear the drawn feature from MapboxDraw to prevent duplicates
      drawRef.current?.deleteAll();

      // Back to Pan after drawing (the effect above switches the draw control)
      setMapTool('pan');
    } finally {
      // Reset the flag after a short delay to allow for any queued events
      setTimeout(() => {
        isProcessingDraw.current = false;
      }, 100);
    }
  };

  // Docked, the pane is at most 640 px wide, so MapLibre opens its compact
  // attribution over the Layers panel; in map mode the full line would run under
  // the legend. Start it compact and collapsed; (i) still opens it.
  const collapseAttribution = () => {
    mapRef.current?.getContainer().querySelector('.maplibregl-ctrl-attrib')?.classList.remove('maplibregl-compact-show');
  };

  return (
    <div className="h-full w-full relative">
      <Map
        ref={mapRef}
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        onMoveEnd={evt => setMapView({ longitude: evt.viewState.longitude, latitude: evt.viewState.latitude, zoom: evt.viewState.zoom })}
        onLoad={() => {
          collapseAttribution();
          // Browser flows (automation only) read map mode's layers and view from here.
          if (!docked && navigator.webdriver) (window as { __dt4lcMap?: unknown }).__dt4lcMap = mapRef.current?.getMap();
          setLoaded(true);
        }}
        mapStyle={mapStyle}
        attributionControl={{ compact: true }}
        style={{ width: '100%', height: '100%' }}
      >
        <RasterAnchor />
        <NavigationControl position="bottom-right" />
        {!docked && <ScaleControl position="bottom-right" />}
        <DrawControl ref={drawRef} onCreate={handleDrawCreate} />

        {/* Render all drawn regions as layers */}
        {drawnRegions.map((region) => {
          const isSelected = selectedRegion?.id === region.id;
          return (
            <Source
              key={region.id}
              id={`region-${region.id}`}
              type="geojson"
              data={region.geometry}
            >
              {/* Fill layer */}
              <Layer
                id={`region-fill-${region.id}`}
                type="fill"
                paint={{
                  'fill-color': isSelected ? colours.regionSelected : colours.region,
                  'fill-opacity': (isSelected ? 0.3 : 0.15) * regionOpacity
                }}
              />
              {/* Outline layer */}
              <Layer
                id={`region-outline-${region.id}`}
                type="line"
                paint={{
                  'line-color': isSelected ? colours.regionSelected : colours.region,
                  'line-width': isSelected ? 3 : 2,
                  'line-opacity': (isSelected ? 0.9 : 0.6) * regionOpacity
                }}
              />
            </Source>
          );
        })}

        <JobLayersManager only={comparing ? compare.left : undefined} />
        <OblastBoundaries />
        {!docked && <RegionLabel />}
      </Map>
      {comparing && <CompareSwipe viewState={viewState} mapStyle={mapStyle} />}
      {!docked && (
        <>
          {/* The toolbar top left and the controls top right; below 1024 px the controls go under the toolbar. */}
          <div className="absolute inset-x-3 top-3 flex items-start justify-between gap-[6px] max-lg:flex-col max-lg:justify-start pointer-events-none">
            <MapToolbar />
            <MapControls map={loaded ? mapRef.current : null} />
          </div>
          <MapLegend />
        </>
      )}
    </div>
  );
}
