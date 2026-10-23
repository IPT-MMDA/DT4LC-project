import { useAppStore } from '../../store/useAppStore';
import { GeoTIFFLayer } from './GeoTIFFLayer';
import { GEETileLayer } from './GEETileLayer';

export function JobLayersManager() {
  const mapLayers = useAppStore((state) => state.mapLayers);

  // Render layers based on type
  return (
    <>
      {mapLayers.map(layer => {
        if (!layer.url || !layer.visible) return null;

        if (layer.type === 'gee-tiles') {
          return (
            <GEETileLayer
              key={layer.id}
              id={layer.id}
              url={layer.url}
              opacity={layer.opacity}
              visible={layer.visible}
            />
          );
        }

        if (layer.type === 'raster') {
          return (
            <GeoTIFFLayer
              key={layer.id}
              id={layer.id}
              url={layer.url}
              opacity={layer.opacity}
              visible={layer.visible}
            />
          );
        }

        return null;
      })}
    </>
  );
}
