import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';
import apiClient from '../api/client';
import { sampleAttachments } from '../api/sample';
import type { Capabilities } from '../api/hooks/useCapabilities';
import { AppRoutes } from '../routes';
import { useAppStore, type DrawnRegion } from '../store/useAppStore';
import { useChatStore } from '../store/useChatStore';
import DEMO_JSON from '../test/fixtures/capabilities/demo-guest.json';
import OFFLINE_JSON from '../test/fixtures/capabilities/offline-member.json';
import { GUEST_SESSION, SAMPLE } from '../test/fixtures';

// Real GET /v1/capabilities: Earth Engine configured (demo) and not configured (offline laptop).
const DEMO = DEMO_JSON as unknown as Capabilities;
const OFFLINE = OFFLINE_JSON as unknown as Capabilities;

// jsdom has no WebGL: the map is a stand-in; the panel is real.
vi.mock('../components/map/MapContainer', () => ({ MapContainer: () => <div data-testid="maplibre-map" /> }));

const HEALTH = { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: false, offline: true, provider: null, model: null } };

const reservoirNorth: DrawnRegion = {
  id: 'region-1',
  name: 'Region 10:00:00',
  type: 'bbox',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[33.27, 46.87], [33.55, 46.87], [33.55, 46.69], [33.27, 46.69], [33.27, 46.87]]] },
  },
  bbox: [33.27, 46.69, 33.55, 46.87],
  createdAt: '2026-10-23T10:00:00.000Z',
};

let post: MockInstance<typeof apiClient.post>;

function mockApi(caps: Capabilities, onPost: (url: string, body: unknown) => unknown = (url) => { throw new Error(`unexpected POST ${url}`); }) {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/capabilities') return caps as never;
    if (url === '/v1/session') return GUEST_SESSION as never;
    if (url === '/v1/health') return HEALTH as never;
    if (url === '/v1/files') return { ok: true, files: [] } as never;
    throw new Error(`unexpected GET ${url}`);
  });
  post = vi.spyOn(apiClient, 'post').mockImplementation(async (url: string, body?: unknown) => onPost(url, body) as never);
}

const location = { current: '' };
function LocationProbe() {
  const loc = useLocation();
  location.current = loc.pathname;
  return null;
}

function renderMap() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/app/map']}>
        <AppRoutes />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const panel = () => within(screen.getByRole('complementary', { name: 'Imagery' }));
const button = (name: string) => panel().getByRole('button', { name });

beforeEach(() => {
  localStorage.clear();
  useAppStore.setState({ drawnRegions: [reservoirNorth], selectedRegion: reservoirNorth, mapLayers: [], imagery: null, regionOpacity: 1, compare: null });
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: sampleAttachments(SAMPLE), pendingRegion: null });
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: (query: string) => ({ matches: false, media: query, addEventListener: () => {}, removeEventListener: () => {} }),
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('map mode imagery panel', () => {
  it('shows the selected region with its area and centroid, and renames it in place', () => {
    mockApi(OFFLINE);
    renderMap();
    const card = panel().getByTestId('region-card');
    expect(within(card).getByText('Region 10:00:00')).toBeTruthy();
    expect(within(card).getByTestId('region-detail').textContent).toBe('427 km² · 46.78 N 33.41 E');

    fireEvent.click(within(card).getByRole('button', { name: 'Rename' }));
    const input = within(card).getByRole('textbox', { name: 'Region name' });
    fireEvent.change(input, { target: { value: 'Reservoir-north' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(useAppStore.getState().selectedRegion?.name).toBe('Reservoir-north');
    expect(useAppStore.getState().drawnRegions[0].name).toBe('Reservoir-north');
  });

  it('the presets add a selectable region each and zoom to it; picking one again reselects it', () => {
    mockApi(OFFLINE);
    renderMap();
    const presets = within(panel().getByRole('group', { name: 'Region presets' }));
    expect(presets.getAllByRole('button').map((b) => b.textContent)).toEqual(['Kakhovka dam', 'Brienz', 'Upload GeoJSON']);

    fireEvent.click(presets.getByRole('button', { name: 'Brienz' }));
    expect(useAppStore.getState().selectedRegion).toMatchObject({ id: 'preset-brienz', name: 'Lake Brienz' });
    expect(useAppStore.getState().fitRequest).toEqual(useAppStore.getState().selectedRegion?.bbox);
    expect(panel().getByTestId('region-detail').textContent).toMatch(/ · 46\.73 N 7\.97 E$/);

    // No sample file here carries a footprint: the box around the dam.
    fireEvent.click(presets.getByRole('button', { name: 'Kakhovka dam' }));
    expect(useAppStore.getState().selectedRegion).toMatchObject({ id: 'preset-kakhovka', name: 'Kakhovka dam' });
    fireEvent.click(presets.getByRole('button', { name: 'Brienz' }));
    expect(useAppStore.getState().drawnRegions.map((r) => r.id)).toEqual(['region-1', 'preset-brienz', 'preset-kakhovka']);
    expect(useAppStore.getState().selectedRegion?.id).toBe('preset-brienz');
    // Presets aren't listed again as other regions.
    expect(panel().queryByRole('button', { name: 'Select Kakhovka dam' })).toBeNull();
    expect(panel().getByRole('button', { name: 'Select Region 10:00:00' })).toBeTruthy();
  });

  it('Upload GeoJSON adds the polygon as the selected region, or says why it cannot', async () => {
    mockApi(OFFLINE);
    renderMap();
    const input = panel().getByLabelText('GeoJSON file') as HTMLInputElement;
    // jsdom's File has no text().
    const geojsonFile = (value: unknown, name: string) => Object.assign(new File([], name), { text: async () => JSON.stringify(value) });
    const point = geojsonFile({ type: 'Point', coordinates: [8, 46.7] }, 'pin.geojson');
    fireEvent.change(input, { target: { files: [point] } });
    expect((await panel().findByTestId('geojson-error')).textContent).toBe('pin.geojson has no Polygon or MultiPolygon. Upload a polygon Feature, FeatureCollection or geometry.');
    expect(useAppStore.getState().selectedRegion?.id).toBe('region-1');

    const shore = { type: 'Feature', properties: { name: 'Bönigen shore' }, geometry: { type: 'Polygon', coordinates: [[[7.9, 46.7], [8, 46.7], [8, 46.75], [7.9, 46.7]]] } };
    fireEvent.change(input, { target: { files: [geojsonFile(shore, 'shore.geojson')] } });
    await waitFor(() => expect(useAppStore.getState().selectedRegion?.name).toBe('Bönigen shore'));
    expect(panel().queryByTestId('geojson-error')).toBeNull();
    expect(within(panel().getByTestId('region-card')).getByText('Bönigen shore')).toBeTruthy();
  });

  it("Fetch to map is disabled with its reason when Earth Engine isn't configured (GET /v1/capabilities)", async () => {
    mockApi(OFFLINE);
    renderMap();
    await waitFor(() => expect(panel().getByTestId('fetch-reason').textContent).toBe("Earth Engine isn't configured on this server"));
    expect((button('Fetch to map') as HTMLButtonElement).disabled).toBe(true);
    // Send to chat works without Earth Engine.
    expect((button('Send to chat') as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(button('Fetch to map'));
    expect(post).not.toHaveBeenCalled();
  });

  it('Send to chat puts dataset, dates, cloud and bands in the composer as the region chip', async () => {
    mockApi(OFFLINE);
    renderMap();
    // The sample is in the composer: its dates are the defaults.
    expect((panel().getByLabelText('Start date') as HTMLInputElement).value).toBe('2023-06-01');
    expect((panel().getByLabelText('End date') as HTMLInputElement).value).toBe('2023-06-09');
    expect(panel().getByTestId('dataset-description').textContent).toBe('NASA HLS L30 (Landsat 8/9) · 30 m · revisit 8 days');

    fireEvent.click(button('Sentinel-2'));
    expect(panel().getByTestId('dataset-description').textContent).toBe('Copernicus Sentinel-2 L2A · 10 m · revisit 5 days');
    fireEvent.click(button('HLS L30'));
    fireEvent.change(panel().getByLabelText('End date'), { target: { value: '2023-06-12' } });
    fireEvent.change(panel().getByRole('slider', { name: 'Max cloud cover' }), { target: { value: '35' } });
    expect(panel().getByTestId('cloud-value').textContent).toBe('35 %');
    fireEvent.click(button('B7 SWIR2'));
    fireEvent.click(button('B2 Blue'));
    expect(button('B2 Blue').getAttribute('aria-pressed')).toBe('false');

    fireEvent.click(button('Send to chat'));
    expect(useChatStore.getState().pendingRegion).toEqual({
      id: 'region-1',
      geometry: reservoirNorth.geometry.geometry,
      name: 'Region 10:00:00',
      dataset: 'HLS L30',
      start_date: '2023-06-01',
      end_date: '2023-06-12',
      cloud_max: 35,
      bands: ['B3', 'B4', 'B5', 'B6', 'B7'],
      area_km2: expect.closeTo(426.7, 1),
    });
    await waitFor(() => expect(location.current).toBe('/app'));
    expect(screen.getByTestId('region-chip').textContent).toContain('Region · Region 10:00:00');
  });

  it('Select all and Clear set the bands; without a band or a region nothing can be sent', () => {
    mockApi(OFFLINE);
    renderMap();
    fireEvent.click(button('Clear'));
    expect((button('Send to chat') as HTMLButtonElement).disabled).toBe(true);
    expect(panel().getByTestId('send-reason').textContent).toBe('Pick at least one band');
    fireEvent.click(button('Select all'));
    expect(useAppStore.getState().imagery?.bands).toEqual(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']);
    act(() => useAppStore.getState().setSelectedRegion(null));
    expect((button('Send to chat') as HTMLButtonElement).disabled).toBe(true);
  });

  it('with Earth Engine configured, Fetch to map bulk-fetches the region and adds the layers', async () => {
    const layers = [
      { tile_url: 'https://ee/tiles/pre/{z}/{x}/{y}', layer_name: 'Landsat 8 - 5 bands - Pre', layer_id: 'gee-pre', period: 'pre', data_type: 'bands', bands: ['SR_B2'] },
      { tile_url: 'https://ee/tiles/post/{z}/{x}/{y}', layer_name: 'Landsat 8 - 5 bands - Post', layer_id: 'gee-post', period: 'post', data_type: 'bands', bands: ['SR_B2'] },
    ];
    mockApi(DEMO, (url) => {
      if (url === '/v1/gee/bulk-fetch') return { ok: true, layers, total_layers: 2, errors: null };
      if (url === '/v1/gee/layers/persist') return { ok: true, layer_id: 'x' };
      throw new Error(`unexpected POST ${url}`);
    });
    renderMap();
    await waitFor(() => expect((button('Fetch to map') as HTMLButtonElement).disabled).toBe(false));
    expect(panel().queryByTestId('fetch-reason')).toBeNull();
    fireEvent.click(button('Fetch to map'));

    await waitFor(() => expect(useAppStore.getState().mapLayers.filter((l) => l.source === 'earth-engine')).toHaveLength(2));
    expect(post).toHaveBeenCalledWith('/v1/gee/bulk-fetch', expect.objectContaining({
      dataset_id: 'landsat-8', bands: ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'], pre_start: '2023-05-29', post_end: '2023-06-12', cloud_cover_max: 20,
    }));
    await waitFor(() => expect(post).toHaveBeenCalledWith('/v1/gee/layers/persist', expect.objectContaining({ layer_id: 'gee-post', start_date: '2023-06-06' })));
    const rows = within(screen.getByRole('region', { name: 'Map layers' })).getAllByTestId('map-layer-row');
    expect(rows[0].textContent).toContain('Landsat 8 - 5 bands - Post');
    expect(rows[0].textContent).toContain('HLS L30 · after · 2023-06-06 → 06-12');
  });

  it('a failed fetch says so in the panel, never silently', async () => {
    mockApi(DEMO, () => {
      throw new Error('No Landsat scene under 20 % cloud between 2023-05-29 and 2023-06-04');
    });
    renderMap();
    await waitFor(() => expect((button('Fetch to map') as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(button('Fetch to map'));
    const alert = await panel().findByRole('alert');
    expect(alert.textContent).toContain("Earth Engine didn't return imagery. No Landsat scene under 20 % cloud");
  });
});

describe('map mode layers', () => {
  it("lists the sample as a true-colour layer under the region, and sets each row's opacity", async () => {
    mockApi(OFFLINE);
    useAppStore.getState().addResultLayer({
      id: 'result-1-difference', name: 'Water change · NDWI', type: 'raster', visible: true, opacity: 0.8, url: '/o/change.tif',
      legend: { kind: 'change', stops: ['#8c4a2f', '#c4cad6', '#2d5f9e'] }, source: 'chat', subtitle: 'from chat', addedAt: new Date().toISOString(),
    });
    renderMap();
    const list = within(screen.getByRole('region', { name: 'Map layers' }));
    await waitFor(() => expect(list.getAllByTestId('map-layer-row')).toHaveLength(3));
    expect(list.getByText('Layers · 3')).toBeTruthy();
    expect(list.getAllByTestId('map-layer-row').map((r) => r.querySelector('div')?.textContent)).toEqual([
      'Water change · NDWIfrom chat · just now',
      'Region 10:00:00region · 427 km²',
      'kakhovka_hls_l30HLS L30 · true colour · 2023-06-09',
    ]);
    // The sample sits under the result on the map.
    expect(useAppStore.getState().mapLayers.map((l) => l.source)).toEqual(['sample', 'chat']);

    fireEvent.change(list.getByRole('slider', { name: 'Opacity of Water change · NDWI' }), { target: { value: '40' } });
    expect(useAppStore.getState().mapLayers[1].opacity).toBe(0.4);
    fireEvent.change(list.getByRole('slider', { name: 'Opacity of Region 10:00:00' }), { target: { value: '55' } });
    expect(useAppStore.getState().regionOpacity).toBe(0.55);
    expect(list.getAllByTestId('map-layer-row')[0].textContent).toMatch(/40$/);

    fireEvent.click(list.getByRole('button', { name: 'Hide kakhovka_hls_l30' }));
    expect(useAppStore.getState().mapLayers[0].visible).toBe(false);
  });
});
