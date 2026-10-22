import { useEffect } from 'react';
import { useJobs } from '../../api/hooks/useJobs';
import { useAppStore } from '../../store/useAppStore';
import { getGeoTIFFOutputsFromJob } from '../../utils/jobGeoTIFF';
import { GeoTIFFLayer } from './GeoTIFFLayer';
import { GEETileLayer } from './GEETileLayer';

export function JobLayersManager() {
  const { data: jobsData } = useJobs({ status: 'completed' });
  const addLayer = useAppStore((state) => state.addLayer);
  const mapLayers = useAppStore((state) => state.mapLayers);

  useEffect(() => {
    if (!jobsData?.jobs) return;

    jobsData.jobs.forEach((job) => {
      const addLayerIfNew = (layerId: string, name: string, url: string) => {
        const exists = mapLayers.find((l) => l.id === layerId);
        if (!exists) {
          addLayer({
            id: layerId,
            name,
            type: 'raster',
            visible: true,
            opacity: 0.7,
            url,
          });
        }
      };

      for (const { path, label, layerId } of getGeoTIFFOutputsFromJob(job)) {
        addLayerIfNew(layerId, label, path);
      }
    });
  }, [jobsData, addLayer, mapLayers]);

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
