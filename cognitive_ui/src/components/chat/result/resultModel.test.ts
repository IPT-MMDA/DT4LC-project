import { describe, expect, it } from 'vitest';
import type { JobRecord } from '../../../api/jobs';
import type { MessageAttachment } from '../../../types';
import bandMathSingle from '../../../test/fixtures/jobs/bandmath-single.json';
import lulc from '../../../test/fixtures/jobs/lulc.json';
import ndmiChange from '../../../test/fixtures/jobs/ndmi-change.json';
import ndviChange from '../../../test/fixtures/jobs/ndvi-change.json';
import ndviSingle from '../../../test/fixtures/jobs/ndvi-single.json';
import ndwiChange from '../../../test/fixtures/jobs/ndwi-change.json';
import ndwiChangeV050 from '../../../test/fixtures/jobs/ndwi-change-v050.json';
import sample from '../../../test/fixtures/jobs/sample.json';
import snow from '../../../test/fixtures/jobs/snow.json';
import statistics from '../../../test/fixtures/jobs/statistics.json';
import {
  dayMonth,
  downloadName,
  formatKm2,
  formatArea,
  formatPercent,
  formatValue,
  prettyFormula,
  resultCardOf,
  sourceLine,
  styleOf,
  type ResultCardModel,
} from './resultModel';

// Fixtures: GET /v1/jobs/{id} on the bundled Kakhovka sample (guest, no assistant),
// trimmed to the artifacts, follow-ups and the planning-ready and done events (the
// full event log in ndwi-change, ndmi-change and statistics), paths rewritten to the
// Docker layout. ndwi-change-v050.json is a job from before the bare
// map PNGs and raster styles (v0.5.0): no `*_bare` refs, no `stats.style`.
const SAMPLE_FILES: MessageAttachment[] = sample.attachments.map((a) => ({
  ...a,
  mime_type: 'image/tiff',
  group: { name: sample.name, line: sample.line },
})) as MessageAttachment[];
const context = { attachments: SAMPLE_FILES };
const card = (job: unknown, title?: string) => resultCardOf(job as JobRecord, context, title) as ResultCardModel;
const only9June = { attachments: [SAMPLE_FILES[1]] };

describe('card selection by artifact kind', () => {
  it('NDWI change: a change card with the water areas, three views and the true dates', () => {
    const c = card(ndwiChange);
    expect(c.kind).toBe('change');
    expect(c.title).toBe('Water change · NDWI');
    expect(c.formula).toBeUndefined();
    if (c.kind !== 'change') throw new Error();
    // Swatches from the change style's ramp: lost → no change → gained.
    expect(c.legend).toEqual([
      { label: 'water gained', tone: 'gained', colour: '#2d5f9e' },
      { label: 'water lost', tone: 'lost', colour: '#8c4a2f' },
      { label: 'no change', tone: 'neutral', colour: '#c4cad6' },
    ]);
    // statistics.water: 13.881 → 170.039 km², 156.761 gained, 0.603 lost (not the full scene's numbers).
    expect(c.stats).toEqual([
      { label: 'Open water before', value: '13.9 km²' },
      { label: 'Open water after', value: '170 km²' },
      { label: 'Gained', value: '+157 km²', tone: 'gained' },
      { label: 'Lost', value: '0.6 km²', tone: 'lost' },
    ]);
    expect(c.views.map((v) => [v.key, v.label, v.overlay])).toEqual([
      ['difference', 'Difference', 'difference · 06-01 → 06-09'],
      ['before', 'Before', 'NDWI before · 2023-06-01'],
      ['after', 'After', 'NDWI after · 2023-06-09'],
    ]);
    // The bare map is shown; the full figure stays the PNG download.
    expect(c.views[0].bare?.url).toMatch(/^\/v1\/download\?path=.*2_ChangeMap_change_map_bare\.png&name=water-change_ndwi_2023-06-01_2023-06-09_map\.png$/);
    expect(c.views[0].image?.url).toMatch(/^\/v1\/download\?path=.*2_ChangeMap_change_map\.png&name=water-change_ndwi_2023-06-01_2023-06-09\.png$/);
    expect(c.views[0].raster?.path).toMatch(/2_ChangeMap_change\.tif$/);
    // Downloads are named readably; the stored paths stay.
    expect(downloadName(c.views[0].raster!)).toBe('water-change_ndwi_2023-06-01_2023-06-09.tif');
    expect(c.views[1].bare?.url).toMatch(/ndwi_before_bare\.png&name=ndwi_2023-06-01_map\.png$/);
    expect(c.views[1].image?.url).toMatch(/ndwi_before\.png&name=ndwi_2023-06-01\.png$/);
    expect(c.views[2].bare?.url).toMatch(/ndwi_after_bare\.png&name=ndwi_2023-06-09_map\.png$/);
    expect(c.views[2].raster?.path).toMatch(/index_after\.tif$/);
    // Layer legends: the styles the tiles are drawn in.
    // The difference layer's legend: the change style and the class areas (class_statistics).
    expect(c.views[0].legend).toEqual({
      kind: 'change',
      stops: ['#8c4a2f', '#c4cad6', '#2d5f9e'],
      range: [-0.5, 0.5],
      classes: [
        { label: 'lost', tone: 'lost', colour: '#8c4a2f', area_km2: 0.603 },
        { label: 'no change', tone: 'neutral', colour: '#c4cad6', area_km2: 373.476 },
        { label: 'gained', tone: 'gained', colour: '#2d5f9e', area_km2: 156.761 },
      ],
    });
    expect(c.views[1].legend).toEqual({
      kind: 'ramp',
      stops: ['#996633', '#ccb280', '#f2f2f2', '#99ccff', '#3380e6', '#003399'],
      range: [-0.5, 0.5],
    });
    expect(c.source).toBe('NASA HLS L30 (Landsat 8/9) · 30 m');
    expect(c.bounds).toEqual([32.388779, 46.455331, 32.691118, 46.663879]);
    expect(c.gpkg).toBeUndefined();
  });

  it('a job from before the bare maps (v0.5.0): the full figures, token swatches, no index ramp', () => {
    const c = card(ndwiChangeV050);
    expect(c.kind).toBe('change');
    if (c.kind !== 'change') throw new Error();
    expect(c.views.map((v) => v.bare)).toEqual([undefined, undefined, undefined]);
    expect(c.views[0].image?.url).toMatch(/2_ChangeMap_change_map\.png$/);
    expect(c.views[1].image?.url).toMatch(/ndwi_before\.png$/);
    // No download_name: the download keeps the stored name.
    expect(downloadName(c.views[0].image!)).toBe('2_ChangeMap_change_map.png');
    expect(c.legend).toEqual([
      { label: 'water gained', tone: 'gained' },
      { label: 'water lost', tone: 'lost' },
      { label: 'no change', tone: 'neutral' },
    ]);
    expect(c.views.map((v) => v.legend)).toEqual([{ kind: 'change' }, undefined, undefined]);
    expect(c.stats.map((s) => s.value)).toEqual(['13.9 km²', '170 km²', '+157 km²', '0.6 km²']);
  });

  it('NDVI change reads statistics.vegetation', () => {
    const c = card(ndviChange);
    expect(c.title).toBe('Vegetation change · NDVI');
    if (c.kind !== 'change') throw new Error();
    expect(c.legend.map((l) => l.label)).toEqual(['vegetation gained', 'vegetation lost', 'no change']);
    expect(c.stats.map((s) => [s.label, s.value])).toEqual([
      ['Vegetation before', '438 km²'],
      ['Vegetation after', '242 km²'],
      ['Gained', '+2.0 km²'],
      ['Lost', '198 km²'],
    ]);
  });

  it('NDMI band-math change: composed, formula in the header, drier / wetter / threshold', () => {
    const c = card(ndmiChange);
    expect(c.kind).toBe('change');
    expect(c.title).toBe('Moisture change · ΔNDMI');
    expect(c.formula).toBe('(B5 − B6)/(B5 + B6) · 06-01 → 06-09 · 30 m');
    if (c.kind !== 'change') throw new Error();
    // areas_km2: 37.927 wetter, 41.961 drier at 0.2.
    expect(c.stats).toEqual([
      { label: 'Drier', value: '42.0 km²', tone: 'lost' },
      { label: 'Wetter', value: '+37.9 km²', tone: 'gained' },
      { label: 'Threshold', value: 'ΔNDMI 0.2' },
    ]);
    expect(c.legend.map((l) => [l.label, l.tone, l.colour])).toEqual([
      ['drier', 'lost', '#8c4a2f'],
      ['wetter', 'gained', '#2d5f9e'],
      ['no change', 'neutral', '#c4cad6'],
    ]);
    expect(c.views.map((v) => Boolean(v.image && v.bare && v.raster))).toEqual([true, true, true]);
    expect(c.views[0].bare?.url).toMatch(/2_BandMathChangeMap_change_map_bare\.png$/);
    expect(c.views[1].bare?.url).toMatch(/index_before_bare\.png$/);
    expect(c.views[1].image?.url).toMatch(/index_before\.png$/);
    // Band-math change is drawn over ±display_range[1]; the layer legend keeps the ramp and that range.
    expect(c.views[0].legend).toEqual({
      kind: 'change',
      stops: ['#8c4a2f', '#c4cad6', '#2d5f9e'],
      range: [-0.41093224674729933, 0.41093224674729933],
      classes: [
        { label: 'drier', tone: 'lost', colour: '#8c4a2f', area_km2: 41.961 },
        { label: 'no change', tone: 'neutral', colour: '#c4cad6', area_km2: 450.94 },
        { label: 'wetter', tone: 'gained', colour: '#2d5f9e', area_km2: 37.927 },
      ],
    });
    expect(c.views[1].legend).toMatchObject({ kind: 'ramp', range: [-1, 1] });
  });

  it('a band-math change of another index uses increase / decrease', () => {
    const savi = structuredClone(ndmiChange) as typeof ndmiChange;
    savi.result.execution.artifacts.BandMathChangeMap.name = 'SAVI';
    const c = card(savi);
    expect(c.title).toBe('SAVI change');
    if (c.kind !== 'change') throw new Error();
    expect(c.stats.map((s) => s.label)).toEqual(['Decrease', 'Increase', 'Threshold']);
    expect(c.legend.map((l) => l.label)).toEqual(['decrease', 'increase', 'no change']);
  });

  it('NDVI single date: an index card titled with the analysed date, ramp and stats', () => {
    const c = card(ndviSingle);
    expect(c.kind).toBe('index');
    // The sample's latest file is the one analysed.
    expect(c.title).toBe('NDVI · 9 June');
    if (c.kind !== 'index') throw new Error();
    // The ramp is the bare map's style, the one its tiles are drawn in.
    expect(c.ramp).toEqual({ range: [-0.2, 0.8], stops: ['#994c1a', '#cc9933', '#ffff66', '#99cc33', '#339933', '#006600'] });
    expect(c.views[0].legend).toEqual({ kind: 'ramp', ...c.ramp });
    expect(c.views[0].bare?.url).toMatch(/1_NDVIMap_ndvi_map_bare\.png$/);
    expect(c.stats).toEqual([
      { label: 'Mean', value: '0.17' },
      { label: 'Min', value: '−1' },
      { label: 'Max', value: '1' },
      { label: 'Valid pixels', value: '589 824' },
    ]);
    expect(c.views[0].image?.url).toMatch(/1_NDVIMap_ndvi_map\.png$/);
    expect(c.views[0].raster?.path).toMatch(/1_NDVIMap_ndvi\.tif$/);
    expect(c.source).toBe('NASA HLS L30 (Landsat 8/9) · 30 m');
    expect(c.bounds).toEqual([32.388779, 46.455331, 32.691118, 46.663879]);
  });

  it('band-math single: an index card with the formula and its display range', () => {
    const c = resultCardOf(bandMathSingle as unknown as JobRecord, only9June) as ResultCardModel;
    expect(c.kind).toBe('index');
    expect(c.title).toBe('NDMI · 9 June');
    expect(c.formula).toBe('(B5 − B6)/(B5 + B6) · 2023-06-09 · 30 m');
    if (c.kind !== 'index') throw new Error();
    expect(c.ramp?.range).toEqual([-1, 1]);
    expect(c.views[0].bare?.url).toMatch(/1_BandMathMap_index_map_bare\.png$/);
    expect(c.views[0].image?.url).toMatch(/1_BandMathMap_index_map\.png$/);
  });

  it('LULC: a classification card with class_statistics colours and areas, empty classes left out', () => {
    const c = card(lulc);
    expect(c.kind).toBe('classification');
    expect(c.title).toBe('Land cover · 9 June');
    if (c.kind !== 'classification') throw new Error();
    expect(c.classes).toEqual([
      { name: 'Water', colour: '#0077be', value: '204 km²' },
      { name: 'Bare Soil', colour: '#c2b280', value: '4.3 km²' },
      { name: 'Sparse Vegetation', colour: '#d4e157', value: '65.6 km²' },
      { name: 'Cropland/Grassland', colour: '#8bc34a', value: '122 km²' },
      { name: 'Dense Vegetation', colour: '#2e7d32', value: '135 km²' },
    ]);
    expect(c.views[0].bare?.url).toMatch(/classification_map_bare\.png$/);
    expect(c.views[0].image?.url).toMatch(/classification_map\.png$/);
    expect(c.views[0].raster?.path).toMatch(/classification\.tif$/);
  });

  it('a class without an area shows its share; one without a colour takes the bare map\'s style', () => {
    const job = structuredClone(lulc) as unknown as { result: { execution: { artifacts: { LULCMap: { class_statistics: Record<string, Record<string, unknown>> } } } } };
    const stats = job.result.execution.artifacts.LULCMap.class_statistics;
    for (const c of Object.values(stats)) c.area_km2 = null;
    delete stats.Water.colour;
    const c = card(job);
    if (c.kind !== 'classification') throw new Error();
    expect(c.classes.slice(0, 2)).toEqual([
      { name: 'Water', colour: '#0077be', value: '38.5 %' },
      { name: 'Bare Soil', colour: '#c2b280', value: '0.8 %' },
    ]);
  });

  it('snow classifier: a classification card from class_statistics', () => {
    const c = card(snow);
    expect(c.title).toBe('Snow and ice · 9 June');
    if (c.kind !== 'classification') throw new Error();
    expect(c.classes).toEqual([
      { name: 'No Snow', colour: '#667f4c', value: '531 km²' },
      { name: 'Snow', colour: '#ffcc66', value: '<0.1 km²' },
    ]);
    expect(c.views[0].bare?.url).toMatch(/snow_classification_bare\.png$/);
    expect(c.views[0].raster?.path).toMatch(/snow_mask\.tif$/);
  });

  it('a snow job from before class_statistics keeps the snow share, without colours', () => {
    const c = card({ result: { execution: { artifacts: { SnowClassification: { statistics: { snow_coverage_percent: 12.5 } } } } } });
    if (c.kind !== 'classification') throw new Error();
    expect(c.classes).toEqual([
      { name: 'Snow / ice', value: '12.5 %' },
      { name: 'Other', value: '87.5 %' },
    ]);
  });

  it('statistics: a per-band table named by the file\'s band codes, no image', () => {
    const c = card(statistics);
    expect(c.kind).toBe('statistics');
    expect(c.title).toBe('Band statistics · 9 June');
    if (c.kind !== 'statistics') throw new Error();
    expect(c.views).toEqual([]);
    expect(c.bands.map((b) => b.band)).toEqual(['B2', 'B3', 'B4', 'B5', 'B6']);
    expect(c.bands[0]).toEqual({ band: 'B2', min: '−1 429', max: '3 533', mean: '333.7', std: '225.9', valid: '589 824' });
  });

  it('statistics from before BR3 (band_<n> keys): named by the file\'s n-th band', () => {
    const old = { result: { execution: { artifacts: { RasterPath: SAMPLE_FILES[1].path, Statistics: { bands: { band_1: { min: 1 }, band_7: { min: 2 } } } } } } };
    const c = card(old);
    if (c.kind !== 'statistics') throw new Error();
    expect(c.bands.map((b) => b.band)).toEqual(['B2', 'Band 7']);
  });

  it('field boundaries: count, area, the overlay PNG and a GeoPackage download', () => {
    // Shape of dta/dti/models/third_party/delineate_anything/inference.py run(); it can't run offline.
    const job = {
      result: {
        execution: {
          artifacts: {
            RasterPath: SAMPLE_FILES[1].path,
            FieldBoundaries: {
              output_path: '/app/resources/.cache/tmp/delineate_outputs/hlsl_20230609_fields.gpkg',
              num_fields: 1284,
              total_area_m2: 487_000_000,
              crs: 'EPSG:32636',
              visualizations: {
                field_boundaries: {
                  kind: 'png',
                  path: '/app/resources/.cache/tmp/delineate_outputs/hlsl_20230609_fields.png',
                  url: '/v1/download?path=%2Fapp%2Fresources%2F.cache%2Ftmp%2Fdelineate_outputs%2Fhlsl_20230609_fields.png',
                  stats: {},
                },
              },
            },
          },
        },
      },
    };
    const c = card(job);
    expect(c.kind).toBe('fields');
    expect(c.title).toBe('Field boundaries · 9 June');
    if (c.kind !== 'fields') throw new Error();
    expect(c.stats).toEqual([
      { label: 'Fields', value: '1 284' },
      { label: 'Total area', value: '487 km²' },
      { label: 'Mean size', value: '37.9 ha' },
    ]);
    expect(c.gpkg?.url).toBe('/v1/download?path=%2Fapp%2Fresources%2F.cache%2Ftmp%2Fdelineate_outputs%2Fhlsl_20230609_fields.gpkg');
    expect(downloadName(c.gpkg!)).toBe('hlsl_20230609_fields.gpkg');
    expect(c.views[0].raster).toBeUndefined();
  });

  it('another step output with an image falls back to an image card; nothing to show is null', () => {
    const png = { kind: 'png', path: '/o/1_Reconstruction_x.png', url: '/v1/download?path=%2Fo%2F1_Reconstruction_x.png', stats: {} };
    const c = card({ result: { execution: { artifacts: { Reconstruction: { visualizations: { x: png } } } } } }, 'Prithvi reconstruction');
    expect(c).toMatchObject({ kind: 'image', title: 'Prithvi reconstruction' });
    expect(card({ result: { intent: 'conversation', response: 'hi' } })).toBeNull();
    expect(card({ result: null })).toBeNull();
  });

  it('ignores the base64 strings of results stored before v0.5.0', () => {
    const old = { result: { execution: { artifacts: { NDVIMap: { index_type: 'ndvi', statistics: { mean: 0.5 }, visualizations: { ndvi_map: 'iVBORw0KGgo' + 'A'.repeat(2000) } } } } } };
    const c = card(old);
    expect(c.kind).toBe('index');
    expect(c.views[0]?.image).toBeUndefined();
  });
});

describe('styleOf', () => {
  const ref = (style: unknown) => ({ kind: 'png', path: '/o/x_bare.png', url: '/v1/download?path=x', stats: { width: 4, height: 4, style } });

  it('reads change, index and class styles', () => {
    expect(styleOf(ref({ kind: 'change', range: [-0.5, 0.5], ramp: ['#8c4a2f', '#c4cad6', '#2d5f9e'] }))).toEqual({
      kind: 'change', range: [-0.5, 0.5], ramp: ['#8c4a2f', '#c4cad6', '#2d5f9e'],
    });
    expect(styleOf(ref({ kind: 'class', classes: { 0: '#667f4c', 1: '#ffcc66' } }))).toEqual({ kind: 'class', classes: { 0: '#667f4c', 1: '#ffcc66' } });
  });

  it('is undefined without a style or with a broken one', () => {
    expect(styleOf(undefined)).toBeUndefined();
    expect(styleOf({ kind: 'png', path: '/o/x.png', url: '/v1/download?path=x', stats: {} })).toBeUndefined();
    expect(styleOf(ref({ kind: 'index', range: [0], ramp: ['#000000', '#ffffff'] }))).toBeUndefined();
    expect(styleOf(ref({ kind: 'index', range: [0, 1], ramp: ['red', 'blue'] }))).toBeUndefined();
  });
});

describe('formatting', () => {
  it('formats km² with one decimal under 100, whole km² from 100, a + sign and grouping', () => {
    expect(formatKm2(13.881)).toBe('13.9 km²');
    expect(formatKm2(0.603)).toBe('0.6 km²');
    expect(formatKm2(170.039)).toBe('170 km²');
    expect(formatKm2(156.761, { sign: true })).toBe('+157 km²');
    expect(formatKm2(0, { sign: true })).toBe('0.0 km²');
    expect(formatKm2(1230.4)).toBe('1 230 km²');
    expect(formatKm2(-4.2)).toBe('−4.2 km²');
    expect(formatKm2(null)).toBe('—');
  });

  it("shows a trace area as <0.1 km², not 0.0 km² (the sample's 15 snow pixels)", () => {
    expect(formatKm2(0.013)).toBe('<0.1 km²');
    expect(formatKm2(0.013, { sign: true })).toBe('<0.1 km²');
    expect(formatKm2(-0.013)).toBe('<0.1 km²');
    expect(formatKm2(0)).toBe('0.0 km²');
    expect(formatKm2(0.06)).toBe('0.1 km²');
  });

  it('formats statistics and shares', () => {
    expect(formatValue(-0.42)).toBe('−0.42');
    expect(formatValue(589824)).toBe('589 824');
    expect(formatValue(1418.83)).toBe('1 419');
    expect(formatValue(undefined)).toBe('—');
    expect(formatPercent(38.48)).toBe('38.5 %');
    expect(formatPercent(0.0025)).toBe('<0.1 %');
  });

  it('sets formulas and dates as the design does', () => {
    expect(prettyFormula('(B5 - B6) / (B5 + B6)')).toBe('(B5 − B6)/(B5 + B6)');
    expect(dayMonth('2023-06-09')).toBe('9 June');
    expect(dayMonth(undefined)).toBe('');
  });
});

describe('source line', () => {
  it('names the sensor in full and adds the resolution', () => {
    expect(sourceLine(context)).toBe('NASA HLS L30 (Landsat 8/9) · 30 m');
  });

  it('joins two sensors, keeps an unknown one as is, and is null when nothing is known', () => {
    const s2 = { id: 'b', filename: 'b.tif', metadata: { sensor: 'Sentinel-2', resolution_m: 10 } };
    const odd = { id: 'c', filename: 'c.tif', metadata: { sensor: 'PlanetScope', resolution_m: 11.12 } };
    expect(sourceLine({ attachments: [SAMPLE_FILES[0], s2] })).toBe('NASA HLS L30 (Landsat 8/9) + Copernicus Sentinel-2 · 30 m');
    expect(sourceLine({ attachments: [odd] })).toBe('PlanetScope · 11.1 m');
    expect(sourceLine({ attachments: [{ id: 'x', filename: 'x.tif' }] })).toBeNull();
  });

  it('uses a region\'s dataset when there are no files', () => {
    const region = {
      id: 'r', geometry: { type: 'Point', coordinates: [0, 0] }, dataset: 'MODIS', start_date: '2023-06-01',
      end_date: '2023-06-09', cloud_max: 20, area_km2: 400,
    } as const;
    expect(sourceLine({ attachments: [], region: { ...region, geometry: { type: 'Point', coordinates: [0, 0] } } })).toBe('NASA MODIS');
  });
});

describe('land-cover cards (models/lulc-change-hazard, models/lulc-forecast, algorithms/lulc-history)', () => {
  const ref = (name: string, style?: unknown) => ({ kind: 'png', path: `/o/${name}.png`, url: `/v1/download?path=/o/${name}.png`, ...(style ? { stats: { style } } : {}) });
  const tif = (name: string) => ({ kind: 'geotiff', path: `/o/${name}.tif`, url: `/v1/download?path=/o/${name}.tif` });
  const classes = { kind: 'class', classes: { '1': '#419bdf', '2': '#397d49', '4': '#e49635', '5': '#c4281b', '8': '#e3e2c3' } };
  const ramp = { kind: 'index', range: [0, 100], ramp: ['#f2efe6', '#5c1a1b'] };
  const jobOf = (key: string, artifact: unknown) => ({ result: { execution: { artifacts: { [key]: artifact } } } });

  it('formats areas in km²: two decimals under 10, one under 100, a trace as <0.01', () => {
    expect([0.001, 0.36, 9.996, 48.31, 1204.4].map((v) => formatArea(v))).toEqual(['<0.01 km²', '0.36 km²', '10.0 km²', '48.3 km²', '1 204 km²']);
    expect(formatArea(-1.45, { sign: true })).toBe('−1.45 km²');
  });

  it('hazards: one view per hazard with its year, a local twin, the start map, and the hazards table', () => {
    const vis: Record<string, unknown> = {};
    const rasters: Record<string, unknown> = {};
    for (const k of ['change_rank', 'change_rank_local', 'rank_forest_loss', 'rank_forest_loss_local']) {
      vis[`${k}_map`] = ref(k);
      vis[`${k}_map_bare`] = ref(`${k}_bare`, ramp);
      rasters[k] = tif(k);
    }
    vis.start_class_map_bare = ref('start_bare', classes);
    rasters.start_class = tif('start_class');
    const c = card(jobOf('ChangeHazardMap', {
      index_type: 'change_rank',
      statistics: {
        valid_pixels: 1200,
        start_class_km2: { Trees: 0.5, Crops: 0.58 },
        hazards: [
          { group: 'any_change', title: 'Any land-cover change', raster: 'change_rank', year: 2028, eligible_km2: 1.08, top_decile_km2: 0.108, top_decile_percent: 10, skill: { auc: 0.8, best_null_auc: 0.73 } },
          { group: 'forest_loss', title: 'Forest loss', raster: 'rank_forest_loss', year: 2027, eligible_km2: 0.5, top_decile_km2: 0.02, top_decile_percent: 4, skill: null },
        ],
        summary: ['Forest loss by 2027: 4 % ...'],
      },
      metadata: { t0: 2025 },
      visualizations: vis,
      rasters,
    }));
    if (c.kind !== 'landcover') throw new Error(c.kind);
    expect(c.title).toBe('Land-cover hazards · from 2025');
    expect(c.views.map((v) => [v.key, v.label])).toEqual([['change_rank', 'Any change'], ['rank_forest_loss', 'Forest loss'], ['start_class', 'Now']]);
    expect(c.views[1].overlay).toBe('rank by 2027 · vs Ukraine');
    expect(c.local.rank_forest_loss.raster?.path).toBe('/o/rank_forest_loss_local.tif');
    expect(c.classes.map((x) => [x.name, x.value, x.colour])).toEqual([['Trees', '0.50 km²', '#397d49'], ['Crops', '0.58 km²', '#e49635']]);
    expect(c.tables[0].rows[0].cells).toEqual(['2028', '1.08 km²', '0.11 km² (10.0 %)', '0.80 vs 0.73']);
  });

  it('an old hazard job without a hazards list stays an index card', () => {
    const c = card(jobOf('ChangeHazardMap', { index_type: 'change_rank', statistics: { mean: 50 }, visualizations: {}, rasters: {} }));
    expect(c.kind).toBe('index');
  });

  it('forecast: the changes first over the faded start map, the condition once, the largest transitions', () => {
    const c = card(jobOf('LandCoverForecast', {
      statistics: {
        t0: 2025, year: 2029, horizon_years: 4, changed_percent: 4.5, changed_km2: 3.31, amounts: 'trend scenario',
        summary_lead: 'If the 2021–2025 trend continues, by 2029:',
        summary: ['Crops would decrease by 24 % (net 1.45 km²); its largest loss is to Rangeland (3.45 km²).'],
        checked: 'Where change lands was checked ...',
        transitions: [{ from: 'Crops', to: 'Rangeland', km2: 3.45, percent_of_from: 5.7 }],
        classes: [{ class: 'Crops', before_km2: 10, after_km2: 8.2, change_km2: -1.8, change_percent: -18 }],
        matrix: { classes: ['Crops', 'Rangeland'], km2: [[8.2, 1.8], [0, 2]], row_percent: [[82, 18], [0, 100]] },
        trend: { pair: [2021, 2025], area_km2: 144, matrix: { classes: ['Crops'], km2: [[10]], row_percent: [[100]] } },
      },
      visualizations: { class_forecast_map_bare: ref('f', classes), changed_map_bare: ref('c', classes), class_start_map_bare: ref('s', classes) },
      rasters: { class_forecast: tif('f'), changed: tif('c'), class_start: tif('s') },
    }));
    if (c.kind !== 'landcover') throw new Error(c.kind);
    expect(c.title).toBe('Land-cover forecast · 2025 → 2029');
    expect(c.note).toBe('scenario: 2021–2025 trend');
    expect(c.views.map((v) => v.label)).toEqual(['Changes', '2029', '2025']);
    expect(c.views[0].base?.path).toBe('/o/s.png');
    expect(c.stats.map((x) => [x.label, x.value])).toEqual([['Changing', '3.31 km² · 4.5 %'], ['Largest change', 'Crops → Rangeland 3.45 km²'], ['Horizon', '4 years']]);
    expect(c.lead).toBe('If the 2021–2025 trend continues, by 2029:');
    expect(c.transitions).toEqual([{ from: 'Crops', to: 'Rangeland', fromColour: '#e49635', toColour: '#e3e2c3', area: '3.45 km²', share: '5.7 % of Crops' }]);
    expect(c.tables[0].rows[0].cells).toEqual(['10.0 km²', '8.20 km²', '−1.80 km²', '−18 %']);
    expect(c.tables[1].rows[0].cells).toEqual(['8.20 km² (82 %)', '1.80 km² (18 %)']);
    expect(c.tables[1].rows[1].cells[0]).toBe('·');
  });

  it('history: periods in the title, area per year in km², an older job in hectares too', () => {
    const base = {
      years: [2017, 2018], periods: { start: [2017, 2019], end: [2023, 2025] },
      changed_percent: 12.5, changed_km2: 2.1, changed_percent_single_years: 20.1, single_year_pair: [2017, 2025],
      classes: [], matrix: { classes: [] }, summary: [],
    };
    const c = card(jobOf('LandCoverHistory', { statistics: { ...base, area_km2: { Crops: [15, 14] } }, visualizations: {}, rasters: {} }));
    if (c.kind !== 'landcover') throw new Error(c.kind);
    expect(c.title).toBe('Land-cover history · 2017–2019 → 2023–2025');
    expect(c.stats.map((s) => [s.label, s.value])).toEqual([['Changed', '2.10 km² · 12.5 %'], ['Largest change', '—'], ['Single years 2017–2025', '20.1 %']]);
    const perYear = (m: ResultCardModel) => (m.kind === 'landcover' ? m.tables.find((t) => t.title === 'Area per year')?.rows : undefined);
    expect(perYear(c)).toEqual([{ label: 'Crops', colour: undefined, cells: ['15.0 km²', '14.0 km²'] }]);
    const old = card(jobOf('LandCoverHistory', { statistics: { ...base, area_ha: { Crops: [1500, 1400] } }, visualizations: {}, rasters: {} }));
    expect(perYear(old)?.[0].cells).toEqual(['15.0 km²', '14.0 km²']);
  });
});
