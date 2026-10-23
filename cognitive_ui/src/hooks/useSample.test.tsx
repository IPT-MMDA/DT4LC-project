import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../api/client';
import { useChatStore } from '../store/useChatStore';
import { SAMPLE } from '../test/fixtures';
import { attachSample, useSampleFromQuery } from './useSample';

function mockSample(response: Promise<unknown> = Promise.resolve(SAMPLE)) {
  return vi.spyOn(apiClient, 'post').mockImplementation(() => response as Promise<never>);
}

function renderAt(url: string) {
  const location = { current: '' };
  const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter initialEntries={[url]}>{children}</MemoryRouter>;
  renderHook(
    () => {
      useSampleFromQuery();
      const loc = useLocation();
      location.current = loc.pathname + loc.search;
    },
    { wrapper },
  );
  return location;
}

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [] });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('attachSample', () => {
  it('puts the pair in the composer as one group, with metadata and mime type', async () => {
    const post = mockSample();
    await attachSample();

    expect(post).toHaveBeenCalledWith('/v1/sample');
    const pending = useChatStore.getState().pendingAttachments;
    expect(pending.map((a) => a.path)).toEqual(['/u/6774daf7_hlsl_20230601.tif', '/u/6774daf7_hlsl_20230609.tif']);
    expect(pending.every((a) => a.group?.name === 'kakhovka_hls_l30' && a.group.line === SAMPLE.line)).toBe(true);
    expect(pending[0]).toMatchObject({ mime_type: 'image/tiff', size_bytes: 4180159, metadata: { date: '2023-06-01' } });
  });
});

describe('?sample=1', () => {
  it('calls the sample API, fills the composer in a new conversation and drops the parameter', async () => {
    useChatStore.setState({ activeConversationId: 'chat_old' });
    const post = mockSample();
    const location = renderAt('/app?sample=1&x=2');

    await waitFor(() => expect(useChatStore.getState().pendingAttachments).toHaveLength(2));
    expect(post).toHaveBeenCalledTimes(1);
    expect(useChatStore.getState().activeConversationId).toBeNull();
    await waitFor(() => expect(location.current).toBe('/app?x=2'));
  });

  it.each(['/app?sample=0', '/app'])('%s does not call it', async (url) => {
    const post = mockSample();
    renderAt(url);
    await act(async () => {});
    expect(post).not.toHaveBeenCalled();
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
  });

  it('leaves the composer empty when the server has no sample', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    mockSample(Promise.reject(new Error('Sample not bundled')));
    const location = renderAt('/app?sample=1');
    await waitFor(() => expect(location.current).toBe('/app'));
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
  });
});
