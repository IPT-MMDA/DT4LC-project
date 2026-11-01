import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../api/client';
import { downloadUrl, type JobSummary } from '../api/library';
import { useChatStore } from '../store/useChatStore';
import { GUEST_SESSION, MEMBER_SESSION } from '../test/fixtures';
import type { SessionInfo } from '../types';
import { Library } from './Library';

const today = (h: number, m: number) => {
  const d = new Date();
  d.setHours(h, m, 0, 0);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00`;
};
const card = { title: 'Water change · NDWI difference', analysis: { id: 'algorithms/change-detection', label: 'Change detection' }, kind: 'change' as const, thumbnail_url: null, duration_s: 48, error: null };
const JOBS: JobSummary[] = [
  { id: 'a1', status: 'completed', prompt: 'Compare water before and after the dam breach', created_at: today(0, 2), card },
  { id: 'b2', status: 'failed', prompt: 'Compare water', created_at: today(0, 1), card: { ...card, title: 'Compare water', duration_s: 12, error: { code: 'EE_NOT_CONFIGURED', message: "Earth Engine isn't configured" } } },
];
const FILES = [{ id: 'up_forest', upload_id: 'up', filename: 'up_forest.tif', path: '/u/up_forest.tif', size: [1, 1], crs: null, bounds: [0, 0, 1, 1], size_bytes: 1, source: 'upload', modified: 1, kind: 'upload', created_at: today(0, 0) }];

function mockApi(session: SessionInfo, jobs = JOBS, files: unknown[] = FILES) {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/session') return session as never;
    throw new Error(`unexpected GET ${url}`);
  });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (url.startsWith('/v1/jobs')) return new Response(JSON.stringify({ jobs, total: jobs.length, limit: 100, offset: 0 }));
      if (url === '/v1/files') return new Response(JSON.stringify({ ok: true, files, count: files.length }));
      throw new Error(`unexpected fetch ${url}`);
    }),
  );
}

function renderLibrary() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Library />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [] });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('Library page', () => {
  it("tells a guest when items expire, from the session's kept_hours", async () => {
    mockApi({ ...GUEST_SESSION, kept_hours: 12 });
    renderLibrary();
    expect(await screen.findByText('Every analysis and file from your conversations. Guest items expire after 12 h.')).toBeTruthy();
  });

  it('drops the expiry for members', async () => {
    mockApi(MEMBER_SESSION);
    renderLibrary();
    await screen.findAllByTestId('library-row');
    expect(screen.getByText('Every analysis and file from your conversations.')).toBeTruthy();
  });

  it('labels a proposal job "Proposal"', async () => {
    mockApi(GUEST_SESSION, [{ id: 'p1', status: 'proposal', prompt: 'Map the flood', created_at: today(0, 3), card: null }], []);
    renderLibrary();
    const pill = await screen.findByTestId('status-pill');
    expect(pill.textContent).toBe('Proposal');
    expect(pill.dataset.status).toBe('proposal');
  });

  it('lists analyses and files, and the filters narrow it', async () => {
    mockApi(GUEST_SESSION);
    renderLibrary();
    expect(await screen.findAllByTestId('library-row')).toHaveLength(3);
    expect(screen.getByTestId('library-count').textContent).toBe('3 items');
    const status = screen.getAllByTestId('library-status').map((s) => s.textContent);
    expect(status).toEqual(['✓Done · 48 s', '!Failed · 12 s', '—']);

    const filters = screen.getByRole('toolbar', { name: 'Filters' });
    fireEvent.click(within(filters).getByRole('button', { name: 'Files' }));
    expect(screen.getAllByTestId('library-name').map((n) => n.textContent)).toEqual(['forest.tif']);
    fireEvent.click(within(filters).getByRole('button', { name: 'All' }));
    fireEvent.click(within(filters).getByRole('button', { name: 'Failed' }));
    expect(screen.getAllByTestId('library-name').map((n) => n.textContent)).toEqual(['Compare water']);
    fireEvent.click(within(filters).getByRole('button', { name: 'Failed' }));

    fireEvent.click(within(filters).getByRole('button', { name: 'Type: any ▾' }));
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'File · upload' }));
    expect(screen.getByTestId('library-count').textContent).toBe('1 item');
    fireEvent.click(within(filters).getByRole('button', { name: 'Type: File · upload ▾' }));
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Any type' }));

    fireEvent.click(within(filters).getByRole('button', { name: 'Last 7 days ▾' }));
    expect(screen.getAllByRole('menuitemradio').map((m) => m.textContent)).toEqual(['Today', 'Last 7 days', 'Last 30 days', 'All time']);
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Today' }));
    expect(screen.getByTestId('library-count').textContent).toBe('3 items');

    // Actions per row: Map for the done change, Retry for the failure, Chat for the file.
    const actions = screen.getAllByTestId('library-row').map((r) => within(r).queryAllByRole('button', { name: /^(Map|Retry|Chat)$/ }).map((b) => b.textContent));
    expect(actions).toEqual([['Map'], ['Retry'], ['Chat']]);
  });

  it('shows "Nothing here yet" with a link to Chat when there is nothing', async () => {
    mockApi(GUEST_SESSION, [], []);
    renderLibrary();
    const empty = await screen.findByTestId('library-empty');
    expect(within(empty).getByText('Nothing here yet')).toBeTruthy();
    expect(within(empty).getByRole('link', { name: 'Go to Chat' }).getAttribute('href')).toBe('/app');
  });

  it('asks before deleting', async () => {
    mockApi(GUEST_SESSION);
    renderLibrary();
    const [row] = await screen.findAllByTestId('library-row');
    fireEvent.click(within(row).getByRole('button', { name: 'More for Water change · NDWI difference' }));
    fireEvent.click(within(row).getByRole('menuitem', { name: 'Delete…' }));
    const confirm = within(row).getByRole('alertdialog', { name: 'Delete Water change · NDWI difference?' });
    fireEvent.click(within(confirm).getByRole('button', { name: 'Keep' }));
    expect(within(row).queryByRole('alertdialog')).toBeNull();
    expect(vi.mocked(fetch).mock.calls.some(([, init]) => (init as RequestInit | undefined)?.method === 'DELETE')).toBe(false);
  });

  it("an export's ↓ downloads it under its file name", async () => {
    const exported = { ...FILES[0], id: 'gee_ndwi', upload_id: null, filename: 'gee_ndwi 2023.tif', path: '/c/gee_exports/gee_ndwi 2023.tif', source: 'export', kind: 'export' };
    mockApi(GUEST_SESSION, [], [exported]);
    renderLibrary();
    const link = await screen.findByRole('link', { name: 'Download gee_ndwi 2023.tif' });
    expect(link.getAttribute('href')).toBe('/v1/download?path=%2Fc%2Fgee_exports%2Fgee_ndwi%202023.tif&name=gee_ndwi%202023.tif');
    expect(link.getAttribute('download')).toBe('gee_ndwi 2023.tif');
    // Without a name the server names it after the stored file.
    expect(downloadUrl('/c/x.tif')).toBe('/v1/download?path=%2Fc%2Fx.tif');
  });
});
