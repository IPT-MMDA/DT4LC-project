import { useEffect } from 'react';
import { useJobs } from '../../api/hooks/useJobs';
import { useAppStore } from '../../store/useAppStore';
import { GeoTIFFLayer } from './GeoTIFFLayer';
import { GEETileLayer } from './GEETileLayer';

export function JobLayersManager() {
  const { data: jobsData } = useJobs({ status: 'completed' });
  const addLayer = useAppStore((state) => state.addLayer);
  const mapLayers = useAppStore((state) => state.mapLayers);

  useEffect(() => {
    if (!jobsData?.jobs) return;

    jobsData.jobs.forEach((job) => {
      // Extract artifacts from job results
      const artifacts = job.result?.execution?.artifacts;
      if (!artifacts) return;

      // Helper to add layer if it doesn't exist
      const addLayerIfNew = (layerId: string, name: string, url: string) => {
        const exists = mapLayers.find(l => l.id === layerId);
        if (!exists) {
          addLayer({
            id: layerId,
            name,
            type: 'raster',
            visible: true,
            opacity: 0.7,
            url
          });
        }
      };

      // NDVI results
      if (artifacts.NDVIMap?.output_path) {
        addLayerIfNew(
          `job-${job.id}-ndvi`,
          `NDVI - ${job.id.slice(0, 8)}`,
          artifacts.NDVIMap.output_path
        );
      }

      // NDWI results
      if (artifacts.NDWIMap?.output_path) {
        addLayerIfNew(
          `job-${job.id}-ndwi`,
          `NDWI - ${job.id.slice(0, 8)}`,
          artifacts.NDWIMap.output_path
        );
      }

      // NDSI results
      if (artifacts.NDSIMap?.output_path) {
        addLayerIfNew(
          `job-${job.id}-ndsi`,
          `NDSI - ${job.id.slice(0, 8)}`,
          artifacts.NDSIMap.output_path
        );
      }

      // Change detection results
      if (artifacts.ChangeMap?.output_path) {
        addLayerIfNew(
          `job-${job.id}-change`,
          `Change - ${job.id.slice(0, 8)}`,
          artifacts.ChangeMap.output_path
        );
      }

      // LULC results
      if (artifacts.LULCMap?.output_path) {
        addLayerIfNew(
          `job-${job.id}-lulc`,
          `LULC - ${job.id.slice(0, 8)}`,
          artifacts.LULCMap.output_path
        );
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
