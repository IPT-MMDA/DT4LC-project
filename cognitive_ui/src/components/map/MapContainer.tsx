import { useRef, useState, useEffect } from 'react';
import Map, { MapRef, NavigationControl, ScaleControl, Source, Layer } from 'react-map-gl/maplibre';
import * as turf from '@turf/turf';
import { useAppStore } from '../../store/useAppStore';
import { DrawControl, DrawControlRef } from './DrawControl';
import { DrawToolbar } from './DrawToolbar';
import { LocationQuickAccess } from './LocationQuickAccess';
import { JobLayersManager } from './JobLayersManager';
import 'maplibre-gl/dist/maplibre-gl.css';

export function MapContainer() {
  const mapRef = useRef<MapRef>(null);
  const drawRef = useRef<DrawControlRef>(null);
  const isProcessingDraw = useRef(false);
  const currentBasemap = useAppStore((state) => state.currentBasemap);
  const addRegion = useAppStore((state) => state.addRegion);
  const drawnRegions = useAppStore((state) => state.drawnRegions);
  const selectedRegion = useAppStore((state) => state.selectedRegion);
  const useNewDatasetPanel = useAppStore((state) => state.useNewDatasetPanel);
  const setMapInstance = useAppStore((state) => state.setMapInstance);
  const [viewState, setViewState] = useState({
    longitude: 33.4,  // Ukraine - Kahovka region
    latitude: 46.8,
    zoom: 10
  });
  const [drawMode, setDrawMode] = useState<'select' | 'draw' | 'delete'>('select');

  // Set map instance in store when map loads
  useEffect(() => {
    if (mapRef.current) {
      setMapInstance(mapRef.current.getMap());
    }
  }, [mapRef.current, setMapInstance]);

  // Handle Escape key to cancel drawing
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && drawMode === 'draw') {
        setDrawMode('select');
        drawRef.current?.changeMode('simple_select');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [drawMode]);

  const handleDrawCreate = (event: any) => {
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

      // Switch back to select mode after drawing
      setDrawMode('select');
      drawRef.current?.changeMode('simple_select');
    } finally {
      // Reset the flag after a short delay to allow for any queued events
      setTimeout(() => {
        isProcessingDraw.current = false;
      }, 100);
    }
  };

  const handleModeChange = (mode: 'select' | 'draw' | 'delete') => {
    setDrawMode(mode);

    if (mode === 'select') {
      drawRef.current?.changeMode('simple_select');
    } else if (mode === 'draw') {
      drawRef.current?.changeMode('draw_polygon');
    } else if (mode === 'delete') {
      const selected = drawRef.current?.getSelected();
      if (selected && selected.features && selected.features.length > 0) {
        drawRef.current?.deleteAll();
      }
      setDrawMode('select');
      drawRef.current?.changeMode('simple_select');
    }
  };

  const handleLocationSelect = (location: { longitude: number; latitude: number; zoom: number }) => {
    mapRef.current?.flyTo({
      center: [location.longitude, location.latitude],
      zoom: location.zoom,
      duration: 2000
    });
  };

  // Basemap style URLs
  // Using free basemaps that don't require API keys
  const basemapStyles: Record<string, any> = {
    streets: 'https://demotiles.maplibre.org/style.json',
    satellite: {
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
    },
    terrain: 'https://demotiles.maplibre.org/style.json'
  };

  return (
    <div className="h-full w-full relative">
      <Map
        ref={mapRef}
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        mapStyle={basemapStyles[currentBasemap] || basemapStyles.streets}
        style={{ width: '100%', height: '100%' }}
      >
        <NavigationControl position="bottom-right" />
        <ScaleControl position="bottom-left" />
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
                  'fill-color': isSelected ? '#3b82f6' : '#10b981',
                  'fill-opacity': isSelected ? 0.3 : 0.15
                }}
              />
              {/* Outline layer */}
              <Layer
                id={`region-outline-${region.id}`}
                type="line"
                paint={{
                  'line-color': isSelected ? '#2563eb' : '#059669',
                  'line-width': isSelected ? 3 : 2,
                  'line-opacity': isSelected ? 0.9 : 0.6
                }}
              />
            </Source>
          );
        })}

        <JobLayersManager />
      </Map>
      <DrawToolbar onModeChange={handleModeChange} currentMode={drawMode} />
      {!useNewDatasetPanel && <LocationQuickAccess onLocationSelect={handleLocationSelect} />}
    </div>
  );
}
