/**
 * Dataset Configuration for Multi-Source Satellite Data Selection
 *
 * Defines metadata for available satellite datasets including bands,
 * supported indices, and collection parameters for Google Earth Engine.
 */

export interface BandConfig {
  id: string;
  name: string;
  description: string;
  wavelength?: string;
  defaultSelected: boolean;
}

export interface DatasetConfig {
  id: string;
  name: string;
  description: string;
  collection: string; // GEE collection ID
  bands: BandConfig[];
  supportedIndices: ('ndvi' | 'ndwi' | 'ndsi')[];
  maxCloudCover: number;
  temporalResolution: string;
  spatialResolution: string;
}

export const DATASETS: Record<string, DatasetConfig> = {
  'sentinel-2': {
    id: 'sentinel-2',
    name: 'Sentinel-2',
    description: 'ESA Copernicus Sentinel-2 Surface Reflectance (10m resolution)',
    collection: 'COPERNICUS/S2_SR_HARMONIZED',
    bands: [
      {
        id: 'B2',
        name: 'Blue',
        description: 'Blue band for water penetration and atmospheric correction',
        wavelength: '490nm',
        defaultSelected: true
      },
      {
        id: 'B3',
        name: 'Green',
        description: 'Green band for vegetation discrimination',
        wavelength: '560nm',
        defaultSelected: true
      },
      {
        id: 'B4',
        name: 'Red',
        description: 'Red band for chlorophyll absorption',
        wavelength: '665nm',
        defaultSelected: true
      },
      {
        id: 'B8',
        name: 'NIR',
        description: 'Near Infrared for vegetation health and biomass',
        wavelength: '842nm',
        defaultSelected: true
      },
      {
        id: 'B11',
        name: 'SWIR1',
        description: 'Shortwave Infrared 1 for moisture content',
        wavelength: '1610nm',
        defaultSelected: true
      },
      {
        id: 'B12',
        name: 'SWIR2',
        description: 'Shortwave Infrared 2 for geology and soil',
        wavelength: '2190nm',
        defaultSelected: false
      }
    ],
    supportedIndices: ['ndvi', 'ndwi', 'ndsi'],
    maxCloudCover: 100,
    temporalResolution: '5 days',
    spatialResolution: '10m'
  },

  'modis': {
    id: 'modis',
    name: 'MODIS Terra/Aqua',
    description: 'NASA MODIS Surface Reflectance 8-Day Composite (250-500m resolution)',
    collection: 'MODIS/006/MOD09A1',
    bands: [
      {
        id: 'sur_refl_b01',
        name: 'Red',
        description: 'Red band (620-670nm) for vegetation monitoring',
        wavelength: '645nm',
        defaultSelected: true
      },
      {
        id: 'sur_refl_b02',
        name: 'NIR',
        description: 'Near Infrared (841-876nm) for vegetation analysis',
        wavelength: '858nm',
        defaultSelected: true
      },
      {
        id: 'sur_refl_b03',
        name: 'Blue',
        description: 'Blue band (459-479nm) for water and atmospheric studies',
        wavelength: '469nm',
        defaultSelected: false
      },
      {
        id: 'sur_refl_b04',
        name: 'Green',
        description: 'Green band (545-565nm) for vegetation discrimination',
        wavelength: '555nm',
        defaultSelected: false
      }
    ],
    supportedIndices: ['ndvi', 'ndwi'],
    maxCloudCover: 100,
    temporalResolution: '8 days (composite)',
    spatialResolution: '250-500m'
  },

  'landsat-8': {
    id: 'landsat-8',
    name: 'Landsat 8/9',
    description: 'USGS Landsat 8/9 Surface Reflectance (30m resolution)',
    collection: 'LANDSAT/LC08/C02/T1_L2',
    bands: [
      {
        id: 'SR_B2',
        name: 'Blue',
        description: 'Blue band for bathymetric mapping',
        wavelength: '482nm',
        defaultSelected: true
      },
      {
        id: 'SR_B3',
        name: 'Green',
        description: 'Green band for vegetation discrimination',
        wavelength: '561nm',
        defaultSelected: true
      },
      {
        id: 'SR_B4',
        name: 'Red',
        description: 'Red band for vegetation discrimination',
        wavelength: '654nm',
        defaultSelected: true
      },
      {
        id: 'SR_B5',
        name: 'NIR',
        description: 'Near Infrared for biomass content',
        wavelength: '865nm',
        defaultSelected: false
      },
      {
        id: 'SR_B6',
        name: 'SWIR1',
        description: 'Shortwave Infrared 1 for moisture content',
        wavelength: '1609nm',
        defaultSelected: false
      },
      {
        id: 'SR_B7',
        name: 'SWIR2',
        description: 'Shortwave Infrared 2 for mineral mapping',
        wavelength: '2201nm',
        defaultSelected: false
      }
    ],
    supportedIndices: ['ndvi', 'ndwi', 'ndsi'],
    maxCloudCover: 100,
    temporalResolution: '16 days',
    spatialResolution: '30m'
  },

  'custom': {
    id: 'custom',
    name: 'Custom Dataset',
    description: 'Upload your own GeoTIFF files for analysis',
    collection: '', // Not applicable for custom uploads
    bands: [], // Detected from uploaded file
    supportedIndices: [],
    maxCloudCover: 0,
    temporalResolution: 'User-defined',
    spatialResolution: 'User-defined'
  }
};

/**
 * Get dataset configuration by ID
 */
export function getDatasetConfig(datasetId: string): DatasetConfig | undefined {
  return DATASETS[datasetId];
}

/**
 * Get all available dataset IDs
 */
export function getAvailableDatasetIds(): string[] {
  return Object.keys(DATASETS);
}

/**
 * Get default selected bands for a dataset
 */
export function getDefaultBands(datasetId: string): string[] {
  const dataset = DATASETS[datasetId];
  if (!dataset) return [];
  return dataset.bands
    .filter(band => band.defaultSelected)
    .map(band => band.id);
}

/**
 * Check if a dataset supports a specific spectral index
 */
export function supportsIndex(
  datasetId: string,
  index: 'ndvi' | 'ndwi' | 'ndsi'
): boolean {
  const dataset = DATASETS[datasetId];
  if (!dataset) return false;
  return dataset.supportedIndices.includes(index);
}
