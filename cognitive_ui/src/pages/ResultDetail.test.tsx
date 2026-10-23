import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { useChatStore } from '../store/useChatStore';
import { GUEST_SESSION } from '../test/fixtures';
import capabilities from '../test/fixtures/capabilities/offline-member.json';
import ndviSingle from '../test/fixtures/jobs/ndvi-single.json';
import ndwiChange from '../test/fixtures/jobs/ndwi-change.json';
import regionFailed from '../test/fixtures/jobs/region-failed.json';
import sample from '../test/fixtures/jobs/sample.json';
import statistics from '../test/fixtures/jobs/statistics.json';
import type { MessageAttachment } from '../types';
import { ResultDetail } from './ResultDetail';

const NDWI = 'Compare water before and after the dam breach';
const CSV = 'class,area_km2,percent,pixels,mean_change\nWater gained,156.761,29.53,174179,0.806\n';
const SAMPLE_FILES = sample.attachments.map((a) => ({ ...a, mime_type: 'image/tiff', group: { name: sample.name, line: sample.line } })) as MessageAttachment[];
const JOBS: Record<string, unknown> = { [ndwiChange.id]: ndwiChange, [statistics.id]: statistics, [ndviSingle.id]: ndviSingle, [regionFailed.id]: regionFailed };

const location = { current: '' };
function Probe() {
  location.current = useLocation().pathname;
  return null;
}

function stubApi() {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/session') return GUEST_SESSION as never;
    if (url === '/v1/capabilities') return capabilities as never;
    throw new Error(`unexpected GET ${url}`);
  });
  const fetchMock = vi.fn(async (url: string) => {
    const csv = /^\/v1\/jobs\/([^/]+)\/statistics\.csv$/.exec(url);
    if (csv) return csv[1] === regionFailed.id ? new Response('{}', { status: 404 }) : new Response(CSV, { headers: { 'content-type': 'text/csv' } });
    const job = /^\/v1\/jobs\/([^/]+)$/.exec(url);
    if (job) {
      const body = JOBS[job[1]];
      return body ? new Response(JSON.stringify(body)) : new Response(JSON.stringify({ ok: false, error: { code: 'not_found', message: `Job ${job[1]} not found` } }), { status: 404 });
    }
    if (url === '/v1/files') return new Response(JSON.stringify({ ok: true, files: [], count: 0 }));
    throw new Error(`unexpected fetch ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

let client: QueryClient;
function renderPage(id: string) {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/app/results/${id}`]}>
        <Routes>
          <Route path="/app/results/:id" element={<ResultDetail />} />
          <Route path="*" element={null} />
        </Routes>
        <Probe />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const texts = (testId: string) => screen.queryAllByTestId(testId).map((e) => e.textContent);

beforeEach(() => {
  localStorage.clear();
  useAppStore.setState({ mapLayers: [], fitRequest: null });
  useChatStore.setState({
    activeConversationId: null,
    pendingAttachments: [],
    draft: '',
    conversations: [
      {
        id: 'k1', title: 'Kakhovka reservoir — water change after breach', createdAt: '', updatedAt: '',
        messages: [
          { id: 'u1', kind: 'user', text: NDWI, attachments: SAMPLE_FILES, createdAt: '' },
          { id: 'a1', kind: 'analysis', jobId: ndwiChange.id, status: 'done', text: '', events: [], createdAt: '' },
          { id: 'u2', kind: 'user', text: 'Give me statistics for every band', createdAt: '' },
          { id: 'a2', kind: 'analysis', jobId: statistics.id, status: 'done', text: '', events: [], createdAt: '' },
          { id: 'u3', kind: 'user', text: ndviSingle.prompt, createdAt: '' },
          { id: 'a3', kind: 'analysis', jobId: ndviSingle.id, status: 'done', text: '', events: [], createdAt: '' },
        ],
      },
    ],
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('result detail of a change', () => {
  it('breadcrumb, header, maps, statistics, plan and downloads from the job', async () => {
    stubApi();
    renderPage(ndwiChange.id);
    await screen.findByTestId('result-detail');

    const crumbs = within(screen.getByTestId('detail-breadcrumb')).getAllByRole('link');
    expect(crumbs.map((a) => [a.textContent, a.getAttribute('href')])).toEqual([
      ['Library', '/app/library'],
      ['Kakhovka reservoir — water change after breach', '/app'],
      ['Water change · NDWI difference', `/app/results/${ndwiChange.id}`],
    ]);
    expect(screen.getByTestId('detail-title').textContent).toBe('Water change · NDWI difference');
    const meta = screen.getByTestId('detail-meta').textContent ?? '';
    expect(meta).toMatch(/^✓Done2 s · (today \d\d:\d\d|yesterday|\d{4}-\d\d-\d\d)/);
    expect(meta.endsWith(ndwiChange.id)).toBe(true);

    expect(screen.getAllByTestId('detail-map').map((m) => m.dataset.view)).toEqual(['difference', 'before', 'after']);
    expect(screen.getAllByTestId('detail-image').map((i) => i.getAttribute('src')?.split('%2F').pop())).toEqual([
      '2_ChangeMap_change_map_bare.png',
      '2_ChangeMap_ndwi_before_bare.png',
      '2_ChangeMap_ndwi_after_bare.png',
    ]);
    expect(texts('detail-map-meta')).toEqual(['2023-06-01 → 2023-06-09 · 30 m · EPSG:32636', '2023-06-01', '2023-06-09']);
    expect(screen.getByRole('region', { name: 'NDWI · before' })).toBeTruthy();
    expect(screen.getByTestId('detail-legend').textContent).toBe('−0.5 · 0 · +0.5 ΔNDWIwater gainedwater lost');
    const difference = screen.getByRole('region', { name: 'Difference map' });
    expect(within(difference).getByRole('link', { name: 'PNG' }).getAttribute('href')).toMatch(/2_ChangeMap_change_map\.png$/);
    expect(within(difference).getByRole('link', { name: 'GeoTIFF' }).getAttribute('href')).toMatch(/2_ChangeMap_change\.tif$/);

    const table = screen.getByTestId('statistics-table');
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Class', 'Area km²', '% of region', 'Mean ΔNDWI']);
    expect(texts('statistics-row')).toEqual(['Water gained156.829.5+0.81', 'Water lost0.60.1−0.23', 'No change373.570.4+0.16']);
    const csv = await within(screen.getByRole('region', { name: 'Statistics' })).findByRole('link', { name: 'CSV' });
    expect(csv.getAttribute('href')).toBe(`/v1/jobs/${ndwiChange.id}/statistics.csv`);

    expect(screen.getByTestId('detail-asked').textContent).toContain(`"${NDWI}"`);
    expect(within(screen.getByTestId('detail-asked')).getByTestId('attachment-chip').textContent).toContain('kakhovka_hls_l30');
    expect(within(screen.getByTestId('detail-plan')).getByRole('heading').textContent).toBe('Plan · 4 steps · 2 s');
    await waitFor(() => expect(texts('plan-step-params')[2]).toBe('(Green − NIR)/(Green + NIR) · after − before · threshold 0.2 · 2 s'));
    expect(screen.getAllByTestId('plan-step').map((s) => s.dataset.status)).toEqual(['done', 'done', 'done', 'done']);
    // No assistant: the "summarization failed" text is no interpretation.
    expect(screen.queryByTestId('interpretation')).toBeNull();

    await waitFor(() => expect(texts('download-row')).toHaveLength(10));
    expect(texts('download-row')[0]).toBe('2_ChangeMap_change.tifGeoTIFF · 2.0 MB');
    expect(texts('download-row')[9]).toBe('statistics.csvCSV · 1 KB');
    expect(screen.getByRole('button', { name: 'Open on map' })).toBeTruthy();
  });

  it('Open on map, Re-run with changes and Open in workspace are wired', async () => {
    stubApi();
    renderPage(ndwiChange.id);
    fireEvent.click(await screen.findByRole('button', { name: 'Re-run with changes' }));
    expect(location.current).toBe('/app');
    expect(useChatStore.getState().draft).toBe(NDWI);
    expect(useChatStore.getState().activeConversationId).toBe('k1');
    cleanup();

    renderPage(ndwiChange.id);
    fireEvent.click(await screen.findByRole('button', { name: 'Open on map' }));
    expect(location.current).toBe('/app/map');
    expect(useAppStore.getState().mapLayers.map((l) => l.id)).toContain(`result-${ndwiChange.id}-difference`);
    cleanup();

    useChatStore.setState({ activeConversationId: null });
    renderPage(ndwiChange.id);
    fireEvent.click(await screen.findByRole('button', { name: 'Open in workspace' }));
    expect(location.current).toBe('/app');
    expect(useChatStore.getState().activeConversationId).toBe('k1');
  });
});

describe('result detail of other kinds', () => {
  it('statistics: a row per band code, no map and no Open on map', async () => {
    stubApi();
    renderPage(statistics.id);
    const table = await screen.findByTestId('statistics-table');
    expect(table.dataset.kind).toBe('bands');
    expect(texts('statistics-row').map((r) => r?.slice(0, 2))).toEqual(['B2', 'B3', 'B4', 'B5', 'B6']);
    expect(screen.queryAllByTestId('detail-map')).toEqual([]);
    expect(screen.queryByRole('button', { name: 'Open on map' })).toBeNull();
    expect(screen.getByTestId('detail-title').textContent).toBe('Raster statistics · 2023-06-09');
  });

  it('a single index: one map and its scalar statistics', async () => {
    stubApi();
    renderPage(ndviSingle.id);
    await screen.findByTestId('statistics-table');
    expect(screen.getAllByTestId('detail-map').map((m) => m.dataset.view)).toEqual(['map']);
    expect(texts('statistics-row')).toEqual(['Mean0.17', 'Min−1', 'Max1', 'Valid pixels589 824']);
  });

  it('a failed region job: header and Asked kept, the failure with its fixes instead of the maps', async () => {
    stubApi();
    renderPage(regionFailed.id);
    const failure = await screen.findByTestId('detail-failure');
    expect(within(failure).getByTestId('failure-code').textContent).toBe('EE_NOT_CONFIGURED');
    expect(within(failure).getAllByRole('button').map((b) => b.textContent)).toEqual(['show raw', 'Use my own image', 'Try the Kakhovka sample']);
    expect(screen.getByTestId('detail-title').textContent).toBe(NDWI);
    expect(screen.getByTestId('status-pill').dataset.status).toBe('failed');
    // No conversation in this browser: the region from the job's echo.
    expect(within(screen.getByTestId('detail-asked')).getByTestId('region-chip').textContent).toContain('Dnipro floodplain');
    expect(screen.queryAllByTestId('detail-map')).toEqual([]);
    expect(screen.queryByTestId('detail-downloads')).toBeNull();
    expect(screen.getByTestId('detail-plan').textContent).toContain('It stopped while planning.');
  });
});

describe('a job that has not finished', () => {
  it('a running job: the plan so far, a link to follow it, and its record dropped on leaving', async () => {
    const running = { ...ndwiChange, id: 'run1', status: 'running', card: null, result: null, events: ndwiChange.events.slice(0, 8) };
    JOBS.run1 = running;
    stubApi();
    const { unmount } = renderPage('run1');
    await screen.findByText('Still running.');
    expect(screen.getByTestId('status-pill').dataset.status).toBe('running');
    expect(screen.getAllByTestId('plan-step').map((s) => s.dataset.status)).toEqual(['done', 'done', 'running', 'queued']);
    expect(screen.queryByRole('button', { name: 'Open on map' })).toBeNull();
    expect(client.getQueryData(['job', 'run1'])).toBeTruthy();
    unmount();
    expect(client.getQueryData(['job', 'run1'])).toBeUndefined();
    delete JOBS.run1;
  });
});

describe('a result that is not available', () => {
  it('an unknown or another owner\'s job (404): links to the Library and Chat', async () => {
    stubApi();
    renderPage('deadbeef');
    const page = await screen.findByTestId('result-unavailable');
    expect(page.textContent).toContain("This result isn't available");
    expect(page.textContent).toContain('It may have expired (guest results are kept 24 h).');
    expect(within(page).getAllByRole('link').map((a) => [a.textContent, a.getAttribute('href')])).toEqual([
      ['Library', '/app/library'],
      ['Chat', '/app'],
    ]);
  });
});
