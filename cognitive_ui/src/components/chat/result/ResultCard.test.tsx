import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { JobRecord } from '../../../api/jobs';
import { useAppStore } from '../../../store/useAppStore';
import { conversationMeta, useChatStore } from '../../../store/useChatStore';
import lulc from '../../../test/fixtures/jobs/lulc.json';
import ndmiChange from '../../../test/fixtures/jobs/ndmi-change.json';
import ndviSingle from '../../../test/fixtures/jobs/ndvi-single.json';
import ndwiChange from '../../../test/fixtures/jobs/ndwi-change.json';
import ndwiChangeV050 from '../../../test/fixtures/jobs/ndwi-change-v050.json';
import sample from '../../../test/fixtures/jobs/sample.json';
import statistics from '../../../test/fixtures/jobs/statistics.json';
import type { MessageAttachment } from '../../../types';
import { ResultCard, ResultCardView } from './ResultCard';
import { resultCardOf, type ResultCardModel } from './resultModel';

const FILES = sample.attachments.map((a) => ({ ...a, group: { name: sample.name, line: sample.line } })) as MessageAttachment[];
const context = { attachments: FILES };
const JOB = 'job-1';
const BOUNDS = [32.388779, 46.455331, 32.691118, 46.663879];
const CHANGE_RAMP = ['#8c4a2f', '#c4cad6', '#2d5f9e'];

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

function setViewport(wide: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: wide && query.includes('min-width: 1440px'),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

function renderIn(ui: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/app']}>
        <Routes>
          <Route path="/app" element={ui} />
          <Route path="/app/map" element={<p>Map page</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const model = (job: unknown) => resultCardOf(job as JobRecord, context) as ResultCardModel;
const renderCard = (job: unknown) => renderIn(<ResultCardView jobId={JOB} model={model(job)} />);
const actions = () => within(screen.getByRole('group', { name: 'Result actions' }));
const actionNames = () =>
  [...screen.getByRole('group', { name: 'Result actions' }).querySelectorAll('button, a')].map((el) => el.textContent);

beforeEach(() => {
  localStorage.clear();
  setViewport(true);
  useAppStore.setState({ mapLayers: [], fitRequest: null });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('change card (NDWI on the sample)', () => {
  it('shows the title, difference image with its overlay, legend, source and the water areas', () => {
    renderCard(ndwiChange);
    const card = screen.getByRole('region', { name: 'Water change · NDWI' });
    expect(card.dataset.kind).toBe('change');
    // The bare map, full-bleed.
    const image = screen.getByTestId('result-image');
    expect(image.getAttribute('src')).toMatch(/^\/v1\/download\?path=.*change_map_bare\.png&name=water-change_ndwi_2023-06-01_2023-06-09_map\.png$/);
    expect(image.className).toContain('object-cover');
    expect(screen.getByTestId('image-overlay').textContent).toBe('difference · 06-01 → 06-09');
    const legend = screen.getByTestId('legend');
    expect(legend.textContent).toBe('water gainedwater lostno change');
    // Swatches in the change style's colours (jsdom reports them as rgb()).
    expect([...legend.querySelectorAll<HTMLElement>('[aria-hidden]')].map((el) => el.style.background)).toEqual([
      'rgb(45, 95, 158)', 'rgb(140, 74, 47)', 'rgb(196, 202, 214)',
    ]);
    expect(screen.getByTestId('source-line').textContent).toBe('NASA HLS L30 (Landsat 8/9) · 30 m');
    const stats = screen.getByTestId('stats');
    expect([...stats.querySelectorAll('dt')].map((d) => d.textContent)).toEqual(['Open water before', 'Open water after', 'Gained', 'Lost']);
    expect([...stats.querySelectorAll('dd')].map((d) => d.textContent)).toEqual(['13.9 km²', '170 km²', '+157 km²', '0.6 km²']);
    expect(stats.querySelectorAll('dd')[2].className).toContain('text-legend-gained');
    expect(stats.querySelectorAll('dd')[3].className).toContain('text-lost');
    expect(screen.queryByTestId('card-formula')).toBeNull();
  });

  it('Difference / Before / After is a segmented control of buttons that switches image, overlay and downloads', () => {
    renderCard(ndwiChange);
    const toggle = within(screen.getByRole('group', { name: 'Result view' }));
    const pressed = () => toggle.getAllByRole('button').map((b) => [b.textContent, b.getAttribute('aria-pressed')]);
    expect(pressed()).toEqual([['Difference', 'true'], ['Before', 'false'], ['After', 'false']]);

    fireEvent.click(toggle.getByRole('button', { name: 'Before' }));
    expect(pressed()).toEqual([['Difference', 'false'], ['Before', 'true'], ['After', 'false']]);
    expect(screen.getByTestId('result-image').getAttribute('src')).toMatch(/ndwi_before_bare\.png&name=ndwi_2023-06-01_map\.png$/);
    expect(screen.getByTestId('image-overlay').textContent).toBe('NDWI before · 2023-06-01');
    expect(actions().getByRole('link', { name: 'GeoTIFF' }).getAttribute('href')).toMatch(/index_before\.tif&name=ndwi_2023-06-01\.tif$/);
    expect(actions().getByRole('link', { name: 'GeoTIFF' }).getAttribute('download')).toBe('ndwi_2023-06-01.tif');
    // PNG is still the full figure, with its title and colour bar.
    expect(actions().getByRole('link', { name: 'PNG' }).getAttribute('href')).toMatch(/ndwi_before\.png&name=ndwi_2023-06-01\.png$/);

    fireEvent.click(toggle.getByRole('button', { name: 'After' }));
    expect(screen.getByTestId('image-overlay').textContent).toBe('NDWI after · 2023-06-09');

    fireEvent.click(toggle.getByRole('button', { name: 'Difference' }));
    expect(screen.getByTestId('image-overlay').textContent).toBe('difference · 06-01 → 06-09');
    expect(actions().getByRole('link', { name: 'GeoTIFF' }).getAttribute('href')).toMatch(/2_ChangeMap_change\.tif&name=water-change_ndwi_2023-06-01_2023-06-09\.tif$/);
  });

  it('offers On map, GeoTIFF, PNG and Details, with download links and no GeoPackage for a raster', () => {
    renderCard(ndwiChange);
    expect(actionNames()).toEqual(['● On map', 'GeoTIFF', 'PNG', 'Details ›']);
    expect(actions().queryByRole('link', { name: 'GeoPackage' })).toBeNull();
    // Saved under the readable download_name, which the tooltip shows too.
    expect(['GeoTIFF', 'PNG'].map((name) => actions().getByRole('link', { name }).getAttribute('download'))).toEqual([
      'water-change_ndwi_2023-06-01_2023-06-09.tif',
      'water-change_ndwi_2023-06-01_2023-06-09.png',
    ]);
    expect(actions().getByRole('link', { name: 'GeoTIFF' }).getAttribute('title')).toBe('water-change_ndwi_2023-06-01_2023-06-09.tif');
    expect(actions().getByRole('link', { name: 'Details ›' }).getAttribute('href')).toBe('/app/results/job-1');
  });

  it('On map (≥1440 px) adds the shown raster as a result layer and asks the map to fit it', () => {
    renderCard(ndwiChange);
    fireEvent.click(actions().getByRole('button', { name: 'On map' }));
    const { mapLayers, fitRequest } = useAppStore.getState();
    expect(mapLayers).toEqual([
      {
        id: 'result-job-1-difference',
        name: 'Water change · NDWI',
        type: 'raster',
        visible: true,
        opacity: 0.8,
        url: expect.stringMatching(/^\/app\/resources\/\.cache\/outputs\/.*2_ChangeMap_change\.tif$/),
        bounds: BOUNDS,
        legend: {
          kind: 'change',
          stops: CHANGE_RAMP,
          range: [-0.5, 0.5],
          classes: [
            { label: 'lost', tone: 'lost', colour: '#8c4a2f', area_km2: 0.603 },
            { label: 'no change', tone: 'neutral', colour: '#c4cad6', area_km2: 373.476 },
            { label: 'gained', tone: 'gained', colour: '#2d5f9e', area_km2: 156.761 },
          ],
        },
        source: 'chat',
        subtitle: 'from chat',
        addedAt: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T/),
      },
    ]);
    expect(fitRequest).toEqual(BOUNDS);
    expect(screen.queryByText('Map page')).toBeNull();

    // Again: the same layer, not a second one; another view is its own layer.
    fireEvent.click(actions().getByRole('button', { name: 'On map' }));
    fireEvent.click(screen.getByRole('button', { name: 'After' }));
    fireEvent.click(actions().getByRole('button', { name: 'On map' }));
    expect(useAppStore.getState().mapLayers.map((l) => [l.id, l.name, l.legend?.kind])).toEqual([
      ['result-job-1-difference', 'Water change · NDWI', 'change'],
      ['result-job-1-after', 'NDWI · 2023-06-09', 'ramp'],
    ]);
  });

  it('On map below 1440 px stores the layer and opens the map', () => {
    setViewport(false);
    renderCard(ndwiChange);
    fireEvent.click(actions().getByRole('button', { name: 'On map' }));
    expect(useAppStore.getState().mapLayers).toHaveLength(1);
    expect(screen.getByText('Map page')).toBeTruthy();
  });
});

describe('older jobs and fallbacks', () => {
  it('a job without bare maps shows the full figure whole, with the legend tokens', () => {
    renderCard(ndwiChangeV050);
    const image = screen.getByTestId('result-image');
    expect(image.getAttribute('src')).toMatch(/2_ChangeMap_change_map\.png$/);
    expect(image.className).toContain('object-contain');
    const swatches = [...screen.getByTestId('legend').querySelectorAll<HTMLElement>('[aria-hidden]')];
    expect(swatches.map((el) => el.className.match(/bg-legend-\S+/)?.[0])).toEqual(['bg-legend-gained', 'bg-legend-lost', 'bg-legend-no-change']);
    expect(actions().getByRole('link', { name: 'PNG' }).getAttribute('href')).toMatch(/2_ChangeMap_change_map\.png$/);
    // No download_name on an older job: the stored name.
    expect(actions().getByRole('link', { name: 'PNG' }).getAttribute('download')).toBe('2_ChangeMap_change_map.png');
  });

  it('PNG offers the bare map when there is no full figure', () => {
    const job = structuredClone(ndwiChange) as unknown as { result: { execution: { artifacts: { ChangeMap: { visualizations: Record<string, unknown> } } } } };
    delete job.result.execution.artifacts.ChangeMap.visualizations.change_map;
    renderCard(job);
    expect(actions().getByRole('link', { name: 'PNG' }).getAttribute('href')).toMatch(/change_map_bare\.png&name=water-change_ndwi_2023-06-01_2023-06-09_map\.png$/);
  });
});

describe('other kinds', () => {
  it('an index card draws its ramp from the bare map\'s style', () => {
    renderCard(ndviSingle);
    expect(screen.getByTestId('legend').textContent).toBe('−0.20.8');
    expect(screen.getByTestId('legend-ramp').style.background).toBe(
      'linear-gradient(90deg, #994c1a, #cc9933, #ffff66, #99cc33, #339933, #006600)',
    );
    expect(screen.getByTestId('result-image').getAttribute('src')).toMatch(/ndvi_map_bare\.png$/);
  });

  it('a classification card lists each class with its colour and area', () => {
    renderCard(lulc);
    const legend = screen.getByTestId('legend');
    expect([...legend.children].map((el) => el.textContent)).toEqual([
      'Water204 km²', 'Bare Soil4.3 km²', 'Sparse Vegetation65.6 km²', 'Cropland/Grassland122 km²', 'Dense Vegetation135 km²',
    ]);
    expect(legend.querySelector<HTMLElement>('[aria-hidden]')?.style.background).toBe('rgb(0, 119, 190)');
  });

  it('a composed change card has the formula in its header', () => {
    renderCard(ndmiChange);
    expect(screen.getByRole('region', { name: 'Moisture change · ΔNDMI' })).toBeTruthy();
    expect(screen.getByTestId('card-formula').textContent).toBe('(B5 − B6)/(B5 + B6) · 06-01 → 06-09 · 30 m');
    expect([...screen.getByTestId('stats').querySelectorAll('dd')].map((d) => d.textContent)).toEqual(['42.0 km²', '+37.9 km²', 'ΔNDMI 0.2']);
    expect(screen.getByTestId('legend').textContent).toBe('drierwetterno change');
  });

  it('statistics: a mono per-band table and only Details', () => {
    renderCard(statistics);
    const rows = within(screen.getByTestId('band-table')).getAllByRole('row').map((r) => r.textContent);
    expect(rows[0]).toBe('BandMinMaxMeanStdValid');
    expect(rows[1]).toBe('B2−1 4293 533333.7225.9589 824');
    expect(rows).toHaveLength(6);
    expect(actionNames()).toEqual(['Details ›']);
  });

  it('field boundaries: the GeoPackage action, and no On map or GeoTIFF without a raster', () => {
    const png = { kind: 'png', path: '/t/f.png', url: '/v1/download?path=%2Ft%2Ff.png', stats: {} };
    renderCard({ result: { execution: { artifacts: { FieldBoundaries: { output_path: '/t/f.gpkg', num_fields: 12, total_area_m2: 1e6, crs: 'EPSG:32636', visualizations: { field_boundaries: png } } } } } });
    expect(actionNames()).toEqual(['PNG', 'GeoPackage', 'Details ›']);
    const gpkg = actions().getByRole('link', { name: 'GeoPackage' });
    expect(gpkg.getAttribute('href')).toBe('/v1/download?path=%2Ft%2Ff.gpkg');
    expect(gpkg.hasAttribute('download')).toBe(true);
  });
});

describe('ResultCard', () => {
  function seed() {
    useChatStore.setState({ conversations: [], activeConversationId: null });
    const { conversationId } = useChatStore.getState().addMessage({ kind: 'user', text: 'Compare water', attachments: FILES });
    useChatStore.getState().addMessage({ kind: 'analysis', jobId: JOB, status: 'done', text: '', events: [], result: { summary: 'Water rose.' } });
    return () => useChatStore.getState().conversations.find((c) => c.id === conversationId)!;
  }

  it('reads the job once and remembers the card kind on the message, so it counts as a result', async () => {
    const conversation = seed();
    const fetchMock = vi.fn(async () => json(200, ndwiChange));
    vi.stubGlobal('fetch', fetchMock);
    expect(conversationMeta(conversation())).toBe('1 file');

    renderIn(<ResultCard jobId={JOB} context={context} />);
    expect(screen.getByTestId('result-loading')).toBeTruthy();
    await screen.findByRole('region', { name: 'Water change · NDWI' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]).toEqual(['/v1/jobs/job-1', expect.objectContaining({ method: 'GET' })]);
    const analysis = conversation().messages[1];
    expect(analysis.kind === 'analysis' && analysis.result).toEqual({ summary: 'Water rose.', card: 'change' });
    expect(conversationMeta(conversation())).toBe('1 file · 1 result');
  });

  it('a result with nothing to show renders no card and is not counted', async () => {
    const conversation = seed();
    vi.stubGlobal('fetch', vi.fn(async () => json(200, { id: JOB, status: 'completed', result: { intent: 'conversation' } })));
    const { container } = renderIn(<ResultCard jobId={JOB} context={context} />);
    await vi.waitFor(() => expect(screen.queryByTestId('result-loading')).toBeNull());
    expect(container.textContent).toBe('');
    expect(conversationMeta(conversation())).toBe('1 file');
  });

  it('a failed read says so and can try again', async () => {
    seed();
    const fetchMock = vi.fn(async () => json(500, { ok: false, error: { code: 'internal', message: 'boom' } }));
    vi.stubGlobal('fetch', fetchMock);
    renderIn(<ResultCard jobId={JOB} context={context} />);
    expect((await screen.findByRole('alert')).textContent).toContain("Couldn't load the result.");
    fetchMock.mockImplementation(async () => json(200, ndwiChange));
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    await screen.findByRole('region', { name: 'Water change · NDWI' });
  });
});
