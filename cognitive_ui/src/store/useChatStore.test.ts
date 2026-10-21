import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ChatMessage, Job, JobResultData } from '../types';

const STORAGE_KEY = 'dt4lc-chat-storage';
const BIG_BASE64 = 'iVBORw0KGgo' + 'A'.repeat(500_000);
const PREVIEW_URL = '/v1/download?path=%2Fapp%2Fuploads%2F21c09d64_scene.png';
const NDVI_URL = '/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F106c8f82%2F1_NDVIMap_ndvi_map.png';

// A fresh module instance creates a fresh store that rehydrates from localStorage,
// which is what a page reload does.
async function loadFreshStore() {
  vi.resetModules();
  return (await import('./useChatStore')).useChatStore;
}

beforeEach(() => {
  localStorage.clear();
});

describe('chat persistence keeps URLs, never image bytes (#67 bug 3)', () => {
  it('persists preview and result URLs, drops base64, and survives a reload', async () => {
    const store = await loadFreshStore();
    // Bytes that reach the store anyway (e.g. an upload response spread into the
    // attachment) must not reach localStorage.
    const attachment = {
      id: '21c09d64',
      filename: 'scene.tif',
      path: '/app/uploads/21c09d64_scene.tif',
      preview_url: PREVIEW_URL,
      preview_png_base64: BIG_BASE64,
    };
    store.getState().addMessage({
      role: 'user',
      content: 'calculate ndvi',
      attachments: [attachment],
    } as ChatMessage);
    const resultData = {
      summary: 'Stub summary.',
      statistics: { type: 'ndvi', values: { mean: '0.0001' } },
      visualizations: [{ type: 'ndvi_map', label: 'NDVI Map', url: NDVI_URL, base64: BIG_BASE64 }],
    } as unknown as JobResultData;
    store.getState().addJobResultMessage({ id: '106c8f82', status: 'completed', progress: 1 } as Job, resultData);

    const raw = localStorage.getItem(STORAGE_KEY) ?? '';
    expect(raw).not.toContain('iVBORw0KGgo');
    expect(raw).not.toContain('base64');
    expect(raw.length).toBeLessThan(5_000);

    const reloaded = await loadFreshStore();
    const [user, result] = reloaded.getState().messages;
    expect(user.attachments).toEqual([
      { id: '21c09d64', filename: 'scene.tif', path: '/app/uploads/21c09d64_scene.tif', preview_url: PREVIEW_URL },
    ]);
    expect(result.resultData?.visualizations).toEqual([{ type: 'ndvi_map', label: 'NDVI Map', url: NDVI_URL }]);
    expect(reloaded.getState().sessions[0].messages[1].resultData?.visualizations?.[0].url).toBe(NDVI_URL);
  });

  it('loads messages persisted by v0.4 (blanked base64, no url) without images', async () => {
    const oldMessages = [
      {
        role: 'user',
        content: 'calculate ndvi',
        attachments: [{ id: 'a1', filename: 'scene.tif', path: '/tmp/a1_scene.tif' }],
      },
      {
        role: 'assistant',
        content: 'Stub summary.',
        type: 'job_result',
        jobId: 'old-job',
        resultData: {
          summary: 'Stub summary.',
          statistics: { type: 'ndvi', values: { mean: '0.4100' } },
          visualizations: [{ type: 'ndvi_map', label: 'NDVI Map', base64: '' }],
        },
      },
    ];
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: {
          sessions: [{ id: 's1', title: 'calculate ndvi', messages: oldMessages, jobIds: ['old-job'], createdAt: '', updatedAt: '' }],
          currentSessionId: 's1',
          messages: oldMessages,
          jobToSessionMap: { 'old-job': 's1' },
        },
        version: 2,
      }),
    );

    const store = await loadFreshStore();
    const [user, result] = store.getState().messages;
    expect(user.attachments?.[0].preview_url).toBeUndefined();
    expect(result.resultData?.summary).toBe('Stub summary.');
    expect(result.resultData?.statistics?.values.mean).toBe('0.4100');
    expect(result.resultData?.visualizations).toEqual([]);
    expect(store.getState().sessions[0].messages[1].resultData?.visualizations).toEqual([]);
    expect(store.getState().getSessionByJobId('old-job')?.id).toBe('s1');
  });
});
