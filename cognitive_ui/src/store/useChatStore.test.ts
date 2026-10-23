import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AnalysisMessage, Attachment, JobResultData, NewChatMessage, UserMessage } from '../types';
import {
  CHAT_STORAGE_KEY,
  CHAT_STORAGE_VERSION,
  findConversationByJobId,
  getContextForBackend,
  selectActiveMessages,
  selectContextTokens,
} from './useChatStore';

const BIG_BASE64 = 'iVBORw0KGgo' + 'A'.repeat(500_000);
const PREVIEW_URL = '/v1/download?path=%2Fapp%2Fuploads%2F21c09d64_scene.png';
const NDVI_URL = '/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F106c8f82%2F1_NDVIMap_ndvi_map.png';

// A fresh module instance creates a fresh store that rehydrates from localStorage,
// which is what a page reload does.
async function loadFreshStore() {
  vi.resetModules();
  return (await import('./useChatStore')).useChatStore;
}

const user = (text: string): NewChatMessage => ({ kind: 'user', text });
const analysis = (jobId: string): NewChatMessage => ({ kind: 'analysis', jobId, status: 'pending', text: '', events: [] });

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('conversations', () => {
  it('starts a conversation on the first message, titled by the first user message', async () => {
    const store = await loadFreshStore();
    expect(store.getState().activeConversationId).toBeNull();

    const { conversationId } = store.getState().addMessage(user('Map moisture change with NDMI between the two dates'));
    store.getState().addMessage(analysis('job-1'));

    const state = store.getState();
    expect(state.activeConversationId).toBe(conversationId);
    expect(state.conversations).toHaveLength(1);
    const [conversation] = state.conversations;
    expect(conversation.title).toBe('Map moisture change with NDMI between the two d...');
    expect(conversation.messages.map((m) => m.kind)).toEqual(['user', 'analysis']);
    expect(conversation.createdAt).toBeTruthy();
    expect(conversation.updatedAt).toBeTruthy();
    expect(selectActiveMessages(state)).toBe(conversation.messages);
  });

  it('keeps the newest conversation first and switches between them', async () => {
    const store = await loadFreshStore();
    const first = store.getState().addMessage(user('first')).conversationId;
    store.getState().newConversation();
    expect(store.getState().activeConversationId).toBeNull();
    expect(selectActiveMessages(store.getState())).toEqual([]);

    const second = store.getState().addMessage(user('second')).conversationId;
    expect(store.getState().conversations.map((c) => c.id)).toEqual([second, first]);

    store.getState().selectConversation(first);
    expect(selectActiveMessages(store.getState()).map((m) => (m as UserMessage).text)).toEqual(['first']);
    store.getState().selectConversation('missing');
    expect(store.getState().activeConversationId).toBe(first);
  });

  it('adds to the given conversation even when another one is active', async () => {
    const store = await loadFreshStore();
    const first = store.getState().addMessage(user('first')).conversationId;
    store.getState().newConversation();
    store.getState().addMessage(user('second'));

    store.getState().addMessage(analysis('job-1'), first);
    const target = store.getState().conversations.find((c) => c.id === first)!;
    expect(target.messages.map((m) => m.kind)).toEqual(['user', 'analysis']);
    expect(store.getState().activeConversationId).not.toBe(first);
  });

  it('renames and deletes conversations', async () => {
    const store = await loadFreshStore();
    const first = store.getState().addMessage(user('first')).conversationId;
    store.getState().newConversation();
    const second = store.getState().addMessage(user('second')).conversationId;

    store.getState().renameConversation(first, 'Kakhovka water');
    expect(store.getState().conversations.find((c) => c.id === first)?.title).toBe('Kakhovka water');

    store.getState().deleteConversation(first);
    expect(store.getState().conversations.map((c) => c.id)).toEqual([second]);
    expect(store.getState().activeConversationId).toBe(second);

    store.getState().deleteConversation(second);
    expect(store.getState().conversations).toEqual([]);
    expect(store.getState().activeConversationId).toBeNull();
  });
});

describe('messages', () => {
  it('updates and removes a message by id', async () => {
    const store = await loadFreshStore();
    const { messageId } = store.getState().addMessage({ kind: 'assistant', text: 'Hi' });
    store.getState().addMessage(user('hello'));

    store.getState().updateMessage(messageId, { text: 'Hello! I map land cover.' });
    expect(selectActiveMessages(store.getState())[0]).toMatchObject({ id: messageId, kind: 'assistant', text: 'Hello! I map land cover.' });

    store.getState().removeMessage(messageId);
    expect(selectActiveMessages(store.getState()).map((m) => m.kind)).toEqual(['user']);
  });

  it('updates an analysis by job id in a conversation that is not active', async () => {
    const store = await loadFreshStore();
    const first = store.getState().addMessage(user('compute ndwi')).conversationId;
    store.getState().addMessage(analysis('job-1'));
    store.getState().newConversation();
    store.getState().addMessage(user('something else'));

    store.getState().appendAnalysisEvent('job-1', { type: 'intent', text: "I'll compute NDWI." });
    store.getState().appendAnalysisEvent('job-1', { type: 'step_started', step: 1, name: 'NDWI' });
    store.getState().updateAnalysis('job-1', { status: 'done', result: { summary: 'Water grew.' } });

    const conversation = findConversationByJobId(store.getState().conversations, 'job-1');
    expect(conversation?.id).toBe(first);
    const message = conversation!.messages[1] as AnalysisMessage;
    expect(message.status).toBe('done');
    expect(message.result?.summary).toBe('Water grew.');
    expect(message.events.map((e) => e.type)).toEqual(['intent', 'step_started']);
    expect(findConversationByJobId(store.getState().conversations, 'unknown')).toBeNull();
  });
});

describe('composer', () => {
  it('holds the draft, pending attachments and pending region, none of them persisted', async () => {
    const store = await loadFreshStore();
    const attachment: Attachment = { id: 'a1', filename: 'scene.tif', path: '/app/uploads/a1_scene.tif' };
    store.getState().setDraft('compute ndvi');
    store.getState().addAttachment(attachment);
    store.getState().addAttachment({ ...attachment, id: 'a2' });
    store.getState().removeAttachment('a2');
    store.getState().setPendingRegion({
      id: 'r1',
      name: 'Region 1',
      type: 'bbox',
      geometry: { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [32.54, 46.56] } },
      bbox: [32.39, 46.46, 32.69, 46.66],
      createdAt: '2023-06-01T00:00:00.000Z',
    });
    store.getState().addMessage(user('compute ndvi'));

    expect(store.getState().draft).toBe('compute ndvi');
    expect(store.getState().pendingAttachments).toEqual([attachment]);
    expect(store.getState().pendingRegion?.id).toBe('r1');

    const persisted = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{}');
    expect(Object.keys(persisted.state).sort()).toEqual(['activeConversationId', 'conversations']);

    store.getState().clearAttachments();
    store.getState().setPendingRegion(null);
    expect(store.getState().pendingAttachments).toEqual([]);
    expect(store.getState().pendingRegion).toBeNull();
  });
});

describe('24 h retention (DESIGN §6)', () => {
  it('drops conversations not updated for 24 h when the page loads', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-10-20T08:00:00Z'));
    const store = await loadFreshStore();
    const old = store.getState().addMessage(user('old')).conversationId;
    store.getState().newConversation();

    vi.setSystemTime(new Date('2026-10-20T20:00:00Z'));
    const recent = store.getState().addMessage(user('recent')).conversationId;
    store.getState().selectConversation(old);

    // 23 h after the old one: both survive a reload.
    vi.setSystemTime(new Date('2026-10-21T07:00:00Z'));
    let reloaded = await loadFreshStore();
    expect(reloaded.getState().conversations.map((c) => c.id)).toEqual([recent, old]);
    expect(reloaded.getState().activeConversationId).toBe(old);

    // 25 h after the old one: it is gone, and so is the selection that pointed at it.
    vi.setSystemTime(new Date('2026-10-21T09:00:00Z'));
    reloaded = await loadFreshStore();
    expect(reloaded.getState().conversations.map((c) => c.id)).toEqual([recent]);
    expect(reloaded.getState().activeConversationId).toBeNull();
  });

  it('counts from the last update, not from creation', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-10-20T08:00:00Z'));
    const store = await loadFreshStore();
    const { conversationId } = store.getState().addMessage(user('compute ndwi'));
    store.getState().addMessage(analysis('job-1'));

    vi.setSystemTime(new Date('2026-10-21T06:00:00Z'));
    store.getState().updateAnalysis('job-1', { status: 'done' });

    vi.setSystemTime(new Date('2026-10-22T05:00:00Z'));
    const reloaded = await loadFreshStore();
    expect(reloaded.getState().conversations.map((c) => c.id)).toEqual([conversationId]);
  });
});

describe('persistence keeps URLs, never image bytes (#67 bug 3)', () => {
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
    store.getState().addMessage({ kind: 'user', text: 'calculate ndvi', attachments: [attachment] });
    store.getState().addMessage(analysis('106c8f82'));
    const result = {
      summary: 'Stub summary.',
      statistics: { type: 'ndvi', values: { mean: '0.0001' } },
      visualizations: [{ type: 'ndvi_map', label: 'NDVI Map', url: NDVI_URL, base64: BIG_BASE64 }],
    } as unknown as JobResultData;
    store.getState().appendAnalysisEvent('106c8f82', { type: 'step_progress', bytes: 1024 });
    store.getState().appendAnalysisEvent('106c8f82', { type: 'done' });
    store.getState().updateAnalysis('106c8f82', { status: 'done', result });

    const raw = localStorage.getItem(CHAT_STORAGE_KEY) ?? '';
    expect(raw).not.toContain('iVBORw0KGgo');
    expect(raw).not.toContain('base64');
    expect(raw.length).toBeLessThan(5_000);

    const reloaded = await loadFreshStore();
    const [sent, done] = selectActiveMessages(reloaded.getState()) as [UserMessage, AnalysisMessage];
    expect(sent.attachments).toEqual([
      { id: '21c09d64', filename: 'scene.tif', path: '/app/uploads/21c09d64_scene.tif', preview_url: PREVIEW_URL },
    ]);
    expect(done.result?.visualizations).toEqual([{ type: 'ndvi_map', label: 'NDVI Map', url: NDVI_URL }]);
    expect(done.events).toEqual([{ type: 'done' }]);
  });
});

describe('migration from v0.5 storage', () => {
  it('drops the v0.5 sessions under the old key instead of crashing, and rewrites the key', async () => {
    const oldMessages = [
      { role: 'user', content: 'calculate ndvi', attachments: [{ id: 'a1', filename: 'scene.tif', path: '/tmp/a1_scene.tif' }] },
      { role: 'assistant', content: 'Processing your request...', type: 'job_submitted', jobId: 'old-job' },
      { role: 'assistant', content: 'Stub summary.', type: 'job_result', jobId: 'old-job', resultData: { summary: 'Stub summary.' } },
    ];
    localStorage.setItem(
      CHAT_STORAGE_KEY,
      JSON.stringify({
        state: {
          sessions: [{ id: 's1', title: 'calculate ndvi', messages: oldMessages, jobIds: ['old-job'], createdAt: '', updatedAt: '' }],
          currentSessionId: 's1',
          messages: oldMessages,
          jobToSessionMap: { 'old-job': 's1' },
        },
        version: 3,
      }),
    );

    const store = await loadFreshStore();
    expect(store.getState().conversations).toEqual([]);
    expect(store.getState().activeConversationId).toBeNull();
    expect(selectContextTokens(store.getState())).toBe(0);

    const persisted = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{}');
    expect(persisted.version).toBe(CHAT_STORAGE_VERSION);
    expect(persisted.state).toEqual({ conversations: [], activeConversationId: null });

    store.getState().addMessage(user('hello'));
    expect(store.getState().conversations).toHaveLength(1);
  });

  it('drops pre-v0.4 storage that has no version', async () => {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ state: { messages: [{ role: 'user', content: 'x' }] } }));
    const store = await loadFreshStore();
    expect(store.getState().conversations).toEqual([]);
  });
});

describe('context counter', () => {
  it('counts ceil(chars / 4) over the active conversation, and survives a reload', async () => {
    const store = await loadFreshStore();
    expect(selectContextTokens(store.getState())).toBe(0);

    store.getState().addMessage(user('a'.repeat(10))); // 10 chars
    expect(selectContextTokens(store.getState())).toBe(3);

    store.getState().addMessage(analysis('job-1')); // no text yet
    expect(selectContextTokens(store.getState())).toBe(3);

    store.getState().updateAnalysis('job-1', { status: 'done', result: { summary: 'b'.repeat(13) } }); // +13
    expect(selectContextTokens(store.getState())).toBe(6); // ceil(23 / 4)

    // The v0.5 counter was a stored number that read 0 after every reload.
    const reloaded = await loadFreshStore();
    expect(selectContextTokens(reloaded.getState())).toBe(6);
  });

  it('follows the active conversation', async () => {
    const store = await loadFreshStore();
    const first = store.getState().addMessage(user('a'.repeat(40))).conversationId;
    store.getState().newConversation();
    expect(selectContextTokens(store.getState())).toBe(0);
    store.getState().addMessage(user('a'.repeat(8)));
    expect(selectContextTokens(store.getState())).toBe(2);
    store.getState().selectConversation(first);
    expect(selectContextTokens(store.getState())).toBe(10);
  });
});

describe('getContextForBackend', () => {
  it('lists earlier attachments with server paths once and summarises the history', async () => {
    const store = await loadFreshStore();
    const att = { id: 'a1', filename: 'scene.tif', path: '/app/uploads/a1_scene.tif' };
    store.getState().addMessage({ kind: 'user', text: 'calculate ndvi', attachments: [att] });
    store.getState().addMessage(analysis('job-1'));
    store.getState().updateAnalysis('job-1', { status: 'done', result: { summary: 'Mean NDVI 0.41.' } });
    store.getState().addMessage({ kind: 'user', text: 'and ndwi', attachments: [att, { id: 'a2', filename: 'no-path.tif' }] });

    expect(getContextForBackend(selectActiveMessages(store.getState()))).toEqual({
      previous_attachments: [{ id: 'a1', filename: 'scene.tif', path: '/app/uploads/a1_scene.tif', mime_type: 'image/tiff' }],
      history: 'User: calculate ndvi\nAssistant: Mean NDVI 0.41.\nUser: and ndwi',
    });
  });
});
