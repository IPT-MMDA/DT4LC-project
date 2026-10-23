import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FILES_QUERY_KEY, useListFiles } from '../../../api/hooks/useUpload';
import { useChatStore } from '../../../store/useChatStore';
import { useMapSample } from './useMapSample';

const group = { name: 'kakhovka_hls_l30.tif', line: 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09 · EPSG:32636' };
const file = (date: string) => ({
  id: `6774daf7_hlsl_${date}`, upload_id: '6774daf7', filename: `hlsl_${date}.tif`, path: `/u/6774daf7_hlsl_${date}.tif`,
  size: [768, 768], crs: 'EPSG:32636', bounds: [0, 0, 1, 1], size_bytes: 4180159, source: 'upload', modified: 1,
  kind: 'sample', group, preview_url: null, created_at: '2026-10-23T10:00:00', metadata: { date },
});
const FILES = { ok: true, count: 2, files: [file('20230601'), file('20230609')] };

beforeEach(() => {
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], pendingRegion: null });
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(FILES))));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useMapSample and From library share GET /v1/files under ['files']", () => {
  it('reads the same cached shape as useListFiles, whichever mounts first', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;

    const picker = renderHook(() => useListFiles(), { wrapper });
    await waitFor(() => expect(picker.result.current.data).toHaveLength(2));
    const map = renderHook(() => useMapSample(), { wrapper });

    expect(map.result.current.map((a) => a.path)).toEqual(['/u/6774daf7_hlsl_20230601.tif', '/u/6774daf7_hlsl_20230609.tif']);
    expect(Array.isArray(client.getQueryData(FILES_QUERY_KEY))).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
