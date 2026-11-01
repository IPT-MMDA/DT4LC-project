import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FILES_QUERY_KEY } from '../../api/hooks/useUpload';
import { JOBS_QUERY_KEY, type JobSummary } from '../../api/library';
import { useAppStore } from '../../store/useAppStore';
import { useChatStore } from '../../store/useChatStore';
import NDWI_JOB from '../../test/fixtures/jobs/ndwi-change.json';
import { libraryRows, type AnalysisRow, type FileRow } from './libraryModel';
import { useLibraryActions } from './useLibraryActions';

const sendMessage = vi.fn<(text: string) => Promise<void>>(async () => {});
vi.mock('../../hooks/useSendMessage', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../hooks/useSendMessage')>()),
  useSendMessage: () => ({ send: sendMessage, sending: false }),
}));

const card = { title: 'Water change · NDWI difference', analysis: { id: 'algorithms/change-detection', label: 'Change detection' }, kind: 'change' as const, thumbnail_url: null, duration_s: 48, error: null };
const NDWI: JobSummary = { id: NDWI_JOB.id, status: 'completed', prompt: 'Compare water before and after the dam breach', created_at: '2026-10-23T14:02:00', card };
const FAILED: JobSummary = { id: 'c2', status: 'failed', prompt: 'Compare water before and after the dam breach', created_at: '2026-10-23T14:06:00', card: { ...card, error: { code: 'EE_NOT_CONFIGURED', message: 'no EE' } } };
const UPLOAD = { id: '35968520_forest', upload_id: '35968520', filename: '35968520_forest.tif', path: '/u/35968520_forest.tif', size: [1, 1] as [number, number], crs: null, bounds: [0, 0, 1, 1] as [number, number, number, number], size_bytes: 7, source: 'upload' as const, modified: 1, kind: 'upload' as const };

const location = { current: '' };
function Probe() {
  location.current = useLocation().pathname;
  return null;
}

let client: QueryClient;
function setup() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/app/library']}>
        {children}
        <Probe />
      </MemoryRouter>
    </QueryClientProvider>
  );
  return renderHook(() => useLibraryActions(), { wrapper }).result;
}

const rowOf = (job: JobSummary) => libraryRows([job], [], useChatStore.getState().conversations)[0] as AnalysisRow;
const fileRow = () => libraryRows([], [UPLOAD], [])[0] as FileRow;

function stubFetch(handler: (url: string, init?: RequestInit) => Response) {
  const mock = vi.fn(async (url: string, init?: RequestInit) => handler(url, init));
  vi.stubGlobal('fetch', mock);
  return mock;
}

beforeEach(() => {
  localStorage.clear();
  sendMessage.mockClear();
  useAppStore.setState({ mapLayers: [], fitRequest: null });
  useChatStore.setState({
    activeConversationId: null,
    pendingAttachments: [],
    conversations: [
      {
        id: 'k1', title: 'Kakhovka reservoir — water change', createdAt: '', updatedAt: '',
        messages: [
          { id: 'u1', kind: 'user', text: NDWI.prompt!, createdAt: '' },
          { id: 'a1', kind: 'analysis', jobId: NDWI.id, status: 'done', text: '', events: [], createdAt: '' },
          { id: 'a2', kind: 'analysis', jobId: 'c2', status: 'failed', text: '', events: [], createdAt: '' },
        ],
      },
      { id: 'k0', title: 'Other', createdAt: '', updatedAt: '', messages: [] },
    ],
  });
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Delete', () => {
  it('deletes a job, then refreshes the jobs and files lists and drops the cached job', async () => {
    const fetchMock = stubFetch(() => new Response(JSON.stringify({ ok: true, id: NDWI.id })));
    const actions = setup();
    client.setQueryData(JOBS_QUERY_KEY, [NDWI]);
    client.setQueryData(FILES_QUERY_KEY, []);
    client.setQueryData(['job', NDWI.id], NDWI_JOB);

    let error: string | null = 'unset';
    await act(async () => {
      error = await actions.current.remove(rowOf(NDWI));
    });

    expect(error).toBeNull();
    expect(fetchMock).toHaveBeenCalledWith(`/v1/jobs/${NDWI.id}`, { method: 'DELETE' });
    expect(client.getQueryState(JOBS_QUERY_KEY)?.isInvalidated).toBe(true);
    expect(client.getQueryState(FILES_QUERY_KEY)?.isInvalidated).toBe(true);
    expect(client.getQueryData(['job', NDWI.id])).toBeUndefined();
  });

  it('deletes a file by its upload id', async () => {
    const fetchMock = stubFetch(() => new Response(JSON.stringify({ ok: true, deleted: ['35968520_forest.tif'] })));
    const actions = setup();
    client.setQueryData(FILES_QUERY_KEY, [UPLOAD]);
    await act(async () => {
      expect(await actions.current.remove(fileRow())).toBeNull();
    });
    expect(fetchMock).toHaveBeenCalledWith('/v1/files/35968520', { method: 'DELETE' });
    expect(client.getQueryState(FILES_QUERY_KEY)?.isInvalidated).toBe(true);
  });

  it('a running job (409) says to cancel it first and keeps the lists', async () => {
    stubFetch(() => new Response(JSON.stringify({ detail: 'Job f5 is running' }), { status: 409 }));
    const actions = setup();
    client.setQueryData(JOBS_QUERY_KEY, [NDWI]);
    await act(async () => {
      expect(await actions.current.remove(rowOf(NDWI))).toBe('Water change · NDWI difference is still running. Cancel it first.');
    });
    expect(client.getQueryState(JOBS_QUERY_KEY)?.isInvalidated).toBe(false);
  });

  it('a 404 (expired or gone) still refreshes the list', async () => {
    stubFetch(() => new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }));
    const actions = setup();
    client.setQueryData(JOBS_QUERY_KEY, [NDWI]);
    await act(async () => {
      expect(await actions.current.remove(rowOf(NDWI))).toBeNull();
    });
    expect(client.getQueryState(JOBS_QUERY_KEY)?.isInvalidated).toBe(true);
  });
});

describe('row actions', () => {
  it('Map adds the result layer from GET /v1/jobs/{id} and opens map mode', async () => {
    stubFetch(() => new Response(JSON.stringify(NDWI_JOB)));
    const actions = setup();
    await act(async () => {
      expect(await actions.current.openOnMap(rowOf(NDWI))).toBeNull();
    });
    expect(useAppStore.getState().mapLayers).toMatchObject([{ id: `result-${NDWI.id}-difference`, type: 'raster', source: 'chat' }]);
    expect(location.current).toBe('/app/map');
  });

  it("Retry resends the prompt in its conversation, with the conversation's files, not the composer's", async () => {
    const actions = setup();
    useChatStore.setState({ pendingAttachments: [{ id: 'x', filename: 'other.tif', path: '/u/other.tif' }] });
    act(() => {
      expect(actions.current.retry(rowOf(FAILED))).toBeNull();
    });
    expect(useChatStore.getState().activeConversationId).toBe('k1');
    expect(sendMessage).toHaveBeenCalledWith('Compare water before and after the dam breach');
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
    expect(location.current).toBe('/app');
  });

  it('Chat opens a new conversation with the file attached', async () => {
    const actions = setup();
    useChatStore.setState({ activeConversationId: 'k1' });
    await act(async () => {
      expect(await actions.current.openInChat(fileRow())).toBeNull();
    });
    expect(useChatStore.getState().activeConversationId).toBeNull();
    expect(useChatStore.getState().pendingAttachments).toMatchObject([{ id: '35968520', path: '/u/35968520_forest.tif', mime_type: 'image/tiff' }]);
    expect(location.current).toBe('/app');
  });

  it('a row click opens Result detail for an analysis', async () => {
    const actions = setup();
    await act(async () => {
      await actions.current.open(rowOf(NDWI));
    });
    expect(location.current).toBe(`/app/results/${NDWI.id}`);
  });
});
