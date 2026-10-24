import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AnalysisMessage, Attachment, JobResultData, NewChatMessage, UserMessage } from '../types';
import {
  CHAT_STORAGE_KEY,
  CHAT_STORAGE_VERSION,
  conversationMeta,
  findConversationByJobId,
  MAX_CONVERSATIONS,
  retainConversations,
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

  it('clearConversations drops every conversation and the composer attachments and region', async () => {
    const store = await loadFreshStore();
    store.getState().addMessage(user('first'));
    store.getState().newConversation();
    store.getState().addMessage(user('second'));
    store.getState().addAttachment({ id: 'a1', filename: 'x.tif', path: '/app/uploads/x.tif' });
    store.getState().setPendingRegion({ name: 'Area 1' } as never);

    store.getState().clearConversations();
    const state = store.getState();
    expect(state.conversations).toEqual([]);
    expect(state.activeConversationId).toBeNull();
    expect(state.pendingAttachments).toEqual([]);
    expect(state.pendingRegion).toBeNull();
    expect(JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{}').state.conversations).toEqual([]);
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
      geometry: { type: 'Polygon', coordinates: [[[32.39, 46.46], [32.69, 46.46], [32.69, 46.66], [32.39, 46.46]]] },
      dataset: 'HLS L30',
      start_date: '2023-06-01',
      end_date: '2023-06-09',
      cloud_max: 20,
      area_km2: 412,
    });
    store.getState().addMessage(user('compute ndvi'));

    expect(store.getState().draft).toBe('compute ndvi');
    expect(store.getState().pendingAttachments).toEqual([attachment]);
    expect(store.getState().pendingRegion?.id).toBe('r1');

    const persisted = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{}');
    expect(Object.keys(persisted.state).sort()).toEqual(['activeConversationId', 'conversations', 'keptHours']);

    store.getState().clearAttachments();
    store.getState().setPendingRegion(null);
    expect(store.getState().pendingAttachments).toEqual([]);
    expect(store.getState().pendingRegion).toBeNull();
  });
});

describe('the sample pair in the composer', () => {
  const group = { name: 'kakhovka_hls_l30', line: 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09' };
  const before: Attachment = { id: '6774daf7', filename: 'hlsl_20230601.tif', path: '/u/6774daf7_hlsl_20230601.tif', group };
  const after: Attachment = { id: '6774daf7', filename: 'hlsl_20230609.tif', path: '/u/6774daf7_hlsl_20230609.tif', group };

  it('keys attachments by path: both files share one id, and attaching again adds nothing', async () => {
    const store = await loadFreshStore();
    store.getState().addAttachment(before);
    store.getState().addAttachment(after);
    store.getState().addAttachment(before);
    store.getState().addAttachment(after);
    expect(store.getState().pendingAttachments).toEqual([before, after]);
  });

  it('keeps the group on a sent message across a reload', async () => {
    const store = await loadFreshStore();
    store.getState().addMessage({ kind: 'user', text: 'Compare water', attachments: [before, after] });
    const reloaded = await loadFreshStore();
    const [sent] = selectActiveMessages(reloaded.getState()) as [UserMessage];
    expect(sent.attachments?.map((a) => a.group)).toEqual([group, group]);
  });
});

describe('conversationMeta', () => {
  it('counts a group once and finished analyses that show a card as results', async () => {
    const store = await loadFreshStore();
    const group = { name: 'kakhovka_hls_l30', line: '' };
    const { conversationId } = store.getState().addMessage({
      kind: 'user',
      text: 'Compare water',
      attachments: [
        { id: 's', filename: 'a.tif', path: '/u/a.tif', group },
        { id: 's', filename: 'b.tif', path: '/u/b.tif', group },
      ],
    });
    const conversation = () => store.getState().conversations.find((c) => c.id === conversationId)!;
    expect(conversationMeta(conversation())).toBe('1 file');

    store.getState().addMessage(analysis('job-1'));
    store.getState().addMessage(analysis('job-2'));
    store.getState().updateAnalysis('job-1', { status: 'done' });
    // Done, but its result card isn't known yet (or it has none).
    expect(conversationMeta(conversation())).toBe('1 file');
    store.getState().updateAnalysis('job-1', { result: { card: 'change' } });
    expect(conversationMeta(conversation())).toBe('1 file · 1 result');

    store.getState().updateAnalysis('job-2', { status: 'done', result: { card: 'index' } });
    store.getState().addMessage({ kind: 'user', text: 'and this', attachments: [{ id: 'c', filename: 'c.tif', path: '/u/c.tif' }] });
    expect(conversationMeta(conversation())).toBe('2 files · 2 results');
  });

  it('is empty for a conversation without files or results', async () => {
    const store = await loadFreshStore();
    store.getState().addMessage(user('hello'));
    expect(conversationMeta(store.getState().conversations[0])).toBe('');
  });
});

describe('retention (DESIGN §6): 24 h before any session is known', () => {
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

describe('retention follows the session', () => {
  it("a guest's conversations expire at the session's kept_hours, also on the next load", async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-10-20T08:00:00Z'));
    const store = await loadFreshStore();
    const old = store.getState().addMessage(user('old')).conversationId;
    store.getState().newConversation();
    vi.setSystemTime(new Date('2026-10-20T12:00:00Z'));
    const recent = store.getState().addMessage(user('recent')).conversationId;
    store.getState().selectConversation(old);

    // A server with GUEST_SESSION_HOURS=6, 5 h after the old one: both kept.
    vi.setSystemTime(new Date('2026-10-20T13:00:00Z'));
    store.getState().setKeptHours(6);
    expect(store.getState().conversations.map((c) => c.id)).toEqual([recent, old]);

    // 7 h after it: the old one goes at once, and with it the selection.
    vi.setSystemTime(new Date('2026-10-20T15:00:00Z'));
    store.getState().setKeptHours(6);
    expect(store.getState().conversations.map((c) => c.id)).toEqual([recent]);
    expect(store.getState().activeConversationId).toBeNull();

    // The next load applies 6 h before the session is fetched again.
    vi.setSystemTime(new Date('2026-10-20T18:30:00Z'));
    const reloaded = await loadFreshStore();
    expect(reloaded.getState().keptHours).toBe(6);
    expect(reloaded.getState().conversations).toEqual([]);
  });

  it("a member's conversations do not expire", async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-10-01T08:00:00Z'));
    const store = await loadFreshStore();
    store.getState().setKeptHours(null);
    const { conversationId } = store.getState().addMessage(user('a month ago'));

    vi.setSystemTime(new Date('2026-10-31T08:00:00Z'));
    store.getState().setKeptHours(null);
    expect(store.getState().conversations.map((c) => c.id)).toEqual([conversationId]);
    const reloaded = await loadFreshStore();
    expect(reloaded.getState().keptHours).toBeNull();
    expect(reloaded.getState().conversations.map((c) => c.id)).toEqual([conversationId]);
    expect(reloaded.getState().activeConversationId).toBe(conversationId);
  });

  it(`keeps at most ${MAX_CONVERSATIONS} conversations, dropping the least recently updated`, async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    const store = await loadFreshStore();
    store.getState().setKeptHours(null);
    const ids: string[] = [];
    for (let i = 0; i < MAX_CONVERSATIONS; i++) {
      vi.setSystemTime(new Date(Date.UTC(2026, 9, 1, 0, i)));
      store.getState().newConversation();
      ids.push(store.getState().addMessage(user(`conversation ${i}`)).conversationId);
    }
    // The oldest one is used again, so the second oldest is the one to go.
    vi.setSystemTime(new Date(Date.UTC(2026, 9, 2)));
    store.getState().addMessage(user('again'), ids[0]);
    vi.setSystemTime(new Date(Date.UTC(2026, 9, 3)));
    store.getState().newConversation();
    const newest = store.getState().addMessage(user('one more')).conversationId;

    const kept = store.getState().conversations.map((c) => c.id);
    expect(kept).toHaveLength(MAX_CONVERSATIONS);
    expect(kept[0]).toBe(newest);
    expect(kept).toContain(ids[0]);
    expect(kept).not.toContain(ids[1]);
  });

  it('retainConversations: the time limit first, then the cap, in list order', () => {
    const at = (id: string, updatedAt: string) => ({ id, title: id, createdAt: updatedAt, updatedAt, messages: [] });
    const list = [at('b', '2026-10-23T10:00:00Z'), at('a', '2026-10-22T09:00:00Z')];
    const now = Date.parse('2026-10-23T10:00:00Z');
    expect(retainConversations(list, 24, now).map((c) => c.id)).toEqual(['b']);
    expect(retainConversations(list, 25, now).map((c) => c.id)).toEqual(['b', 'a']);
    expect(retainConversations(list, null, now).map((c) => c.id)).toEqual(['b', 'a']);
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
    expect(persisted.state).toEqual({ conversations: [], activeConversationId: null, keptHours: 24 });

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

describe('what a follow-up needs survives a reload', () => {
  it("keeps the conversation's pending and the sent files' metadata and MIME type", async () => {
    const store = await loadFreshStore();
    const metadata = { sensor: 'HLS L30', date: '2023-06-09', line: 'HLS L30 · B2–B6 · 30 m · 2023-06-09' };
    const pending = { prompt: 'Map vegetation health', item_id: null, needs: ['RasterPath'], asked_at: '2026-10-23T07:00:00Z' };
    const { conversationId } = store.getState().addMessage({
      kind: 'user',
      text: 'here',
      attachments: [{ id: 'k', filename: 'hlsl_20230609.tif', path: '/u/k.tif', mime_type: 'image/tiff', size_bytes: 4, metadata }],
    });
    store.getState().setConversationPending(conversationId, pending);

    const reloaded = await loadFreshStore();
    const conversation = reloaded.getState().conversations[0];
    expect(conversation.pending).toEqual(pending);
    expect((conversation.messages[0] as UserMessage).attachments?.[0]).toMatchObject({ mime_type: 'image/tiff', size_bytes: 4, metadata });

    reloaded.getState().setConversationPending(conversationId, null);
    expect(reloaded.getState().conversations[0].pending).toBeNull();
  });

  it('titles a conversation by the first user message with text', async () => {
    const store = await loadFreshStore();
    store.getState().addMessage({ kind: 'user', text: '', attachments: [{ id: 'k', filename: 'a.tif', path: '/u/a.tif' }] });
    expect(store.getState().conversations[0].title).toBe('New chat');
    store.getState().addMessage({ kind: 'user', text: 'Map open water' });
    expect(store.getState().conversations[0].title).toBe('Map open water');
  });
});
