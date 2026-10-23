import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { sampleAttachments } from '../api/sample';
import { useChatStore, selectActiveMessages } from '../store/useChatStore';
import { SAMPLE } from '../test/fixtures';
import type { AnalysisMessage, AssistantMessage, ChatRegion, UserMessage } from '../types';
import { conversationHistory, sendMessage, useSendMessage } from './useSendMessage';

// POST /v1/chat answers as the server does (server/routes/chat.py _frames).
function sse(frames: [string, unknown][]): Response {
  const text = frames.map(([event, data], i) => `id: ${i + 1}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('');
  const bytes = new TextEncoder().encode(text);
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      // Frames split mid-line, as a network delivers them.
      for (let i = 0; i < bytes.length; i += 17) controller.enqueue(bytes.slice(i, i + 17));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
}

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const PENDING = { prompt: 'Map vegetation health', item_id: 'algorithms/ndvi', needs: ['RasterPath'], asked_at: '2026-10-23T07:00:00Z' };
const INTENT = "I'll compute NDWI for 1 June and 9 June from the sample, then difference them to map where water changed.";
const JOB = {
  id: 'job-1',
  status: 'pending',
  events: [{ seq: 1, type: 'intent', data: { text: INTENT, t: 1 } }],
};

let fetchMock: ReturnType<typeof vi.fn>;
const bodies = () => fetchMock.mock.calls.map(([, init]) => JSON.parse((init as RequestInit).body as string));
const messages = () => selectActiveMessages(useChatStore.getState());
const conversation = () => useChatStore.getState().conversations.find((c) => c.id === useChatStore.getState().activeConversationId);

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], pendingRegion: null, draft: '' });
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('sendMessage', () => {
  it('posts SSE to /v1/chat and streams reply_delta into one assistant message', async () => {
    fetchMock.mockResolvedValueOnce(
      sse([
        ['reply_delta', { text: 'Hi! I can **map** ' }],
        ['reply_delta', { text: 'water and vegetation.' }],
        ['reply_done', { text: 'Hi! I can **map** water and vegetation.' }],
      ]),
    );
    const seen: string[] = [];
    const unsubscribe = useChatStore.subscribe((s) => {
      const last = selectActiveMessages(s).at(-1);
      if (last?.kind === 'assistant' && seen.at(-1) !== last.text) seen.push(last.text);
    });
    useChatStore.getState().setDraft('hello');

    await sendMessage('hello');
    unsubscribe();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/v1/chat');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>).Accept).toBe('text/event-stream');
    expect(bodies()[0]).toEqual({ message: 'hello', history: [], attachments: [] });

    expect(seen).toEqual(['Hi! I can **map** ', 'Hi! I can **map** water and vegetation.']);
    const [user, reply] = messages() as [UserMessage, AssistantMessage];
    expect(user).toMatchObject({ kind: 'user', text: 'hello' });
    expect(reply).toMatchObject({ kind: 'assistant', text: 'Hi! I can **map** water and vegetation.' });
    expect(messages()).toHaveLength(2);
    expect(conversation()?.title).toBe('hello');
    expect(useChatStore.getState().draft).toBe('');
  });

  it('stores pending from reply_done, sends it back with the files, and clears it on the analysis', async () => {
    fetchMock
      .mockResolvedValueOnce(sse([['reply_done', { text: 'Attach a GeoTIFF and I will run it.', pending: PENDING }]]))
      .mockResolvedValueOnce(sse([['analysis', { job_id: 'job-1', job: JOB }]]));

    await sendMessage('Map vegetation health');
    expect(conversation()?.pending).toEqual(PENDING);

    sampleAttachments(SAMPLE).forEach(useChatStore.getState().addAttachment);
    await sendMessage('here');

    const second = bodies()[1];
    expect(second.pending).toEqual(PENDING);
    expect(second.message).toBe('here');
    expect(second.attachments).toHaveLength(2);
    expect(second.attachments[0]).toEqual({
      id: '6774daf7',
      filename: 'hlsl_20230601.tif',
      path: '/u/6774daf7_hlsl_20230601.tif',
      mime_type: 'image/tiff',
      size_bytes: 4180159,
      metadata: SAMPLE.attachments[0].metadata,
    });
    expect(second.history).toEqual([
      { role: 'user', text: 'Map vegetation health' },
      { role: 'assistant', text: 'Attach a GeoTIFF and I will run it.' },
    ]);
    expect(conversation()?.pending).toBeNull();
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
  });

  it('a reply without pending clears the stored one', async () => {
    fetchMock
      .mockResolvedValueOnce(sse([['reply_done', { text: 'Attach a file.', pending: PENDING }]]))
      .mockResolvedValueOnce(sse([['reply_done', { text: 'Hello!' }]]));
    await sendMessage('Map vegetation health');
    await sendMessage('hello');
    expect(bodies()[1].pending).toEqual(PENDING);
    expect(conversation()?.pending).toBeNull();
  });

  it('adds one analysis message on `analysis` and re-sends the files with follow-ups', async () => {
    const onQuotaChanged = vi.fn();
    fetchMock
      .mockResolvedValueOnce(sse([['analysis', { job_id: 'job-1', job: JOB }]]))
      .mockResolvedValueOnce(sse([['reply_done', { text: 'NDWI uses green and NIR.' }]]));
    sampleAttachments(SAMPLE).forEach(useChatStore.getState().addAttachment);

    await sendMessage('Compare water before and after the dam breach', { onQuotaChanged });

    const [user, analysis] = messages() as [UserMessage, AnalysisMessage];
    expect(user.attachments?.map((a) => a.filename)).toEqual(['hlsl_20230601.tif', 'hlsl_20230609.tif']);
    expect(analysis).toMatchObject({ kind: 'analysis', jobId: 'job-1', status: 'pending', text: '', events: JOB.events });
    expect(onQuotaChanged).toHaveBeenCalledTimes(1);

    // The follow-up has no files in the composer: the conversation's files go again.
    useChatStore.getState().updateAnalysis('job-1', {
      status: 'done',
      events: [...JOB.events, { seq: 9, type: 'done', data: { summary: '3 steps · 48 s · NDWI', t: 2 } }],
    });
    await sendMessage('Why is NDWI better than NDVI for water?');
    const followUp = bodies()[1];
    expect(followUp.attachments.map((a: { path: string }) => a.path)).toEqual([
      '/u/6774daf7_hlsl_20230601.tif',
      '/u/6774daf7_hlsl_20230609.tif',
    ]);
    expect(followUp.history).toEqual([
      { role: 'user', text: 'Compare water before and after the dam breach' },
      { role: 'assistant', text: `${INTENT} Done: 3 steps · 48 s · NDWI` },
    ]);
    // The follow-up's own message shows no chips: the files were sent before.
    expect((messages()[2] as UserMessage).attachments).toBeUndefined();
  });

  it('sends a region instead of the earlier files, and takes the echoed line from the job', async () => {
    const region: ChatRegion = {
      id: 'region-1',
      geometry: { type: 'Polygon', coordinates: [[[32.5, 46.5], [32.6, 46.5], [32.6, 46.6], [32.5, 46.5]]] },
      name: 'Reservoir-north',
      dataset: 'HLS L30',
      start_date: '2023-06-01',
      end_date: '2023-06-09',
      cloud_max: 20,
      area_km2: 42.4,
    };
    const line = 'Region · Reservoir-north · 43 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud';
    fetchMock
      .mockResolvedValueOnce(sse([['reply_done', { text: 'ok' }]]))
      .mockResolvedValueOnce(sse([['analysis', { job_id: 'job-2', job: { ...JOB, id: 'job-2', region: { line, area_km2: 43.1 } } }]]));
    sampleAttachments(SAMPLE).forEach(useChatStore.getState().addAttachment);
    await sendMessage('hello');

    useChatStore.getState().setPendingRegion(region);
    await sendMessage('Map open water with NDWI');

    const body = bodies()[1];
    expect(body.attachments).toEqual([]);
    expect(body.region).toEqual({
      geometry: region.geometry,
      name: 'Reservoir-north',
      dataset: 'HLS L30',
      start_date: '2023-06-01',
      end_date: '2023-06-09',
      cloud_max: 20,
    });
    const sent = messages()[2] as UserMessage;
    expect(sent.region).toMatchObject({ id: 'region-1', line, area_km2: 43.1 });
    expect(useChatStore.getState().pendingRegion).toBeNull();
  });

  it('shows the guest quota refusal inline, with its reset time', async () => {
    const onQuotaChanged = vi.fn();
    const message = 'Guests can run 5 analyses per hour on this server. The next one is available at 04:21 UTC.';
    fetchMock.mockResolvedValueOnce(
      json(429, { ok: false, error: { code: 'GUEST_QUOTA', message, details: { limit: 5, used: 5, resets_at: '2026-10-23T04:21:00Z' } } }),
    );
    await sendMessage('Compare water before and after the dam breach', { onQuotaChanged });
    expect(messages().at(-1)).toMatchObject({ kind: 'assistant', text: message, error: true });
    expect(onQuotaChanged).toHaveBeenCalled();
  });

  it('shows other errors as a short inline error', async () => {
    fetchMock
      .mockResolvedValueOnce(json(422, { detail: [{ msg: 'region start_date is after end_date' }] }))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await sendMessage('one');
    await sendMessage('two');
    const errors = messages().filter((m) => m.kind === 'assistant');
    expect(errors.map((m) => [m.text, (m as AssistantMessage).error])).toEqual([
      ["Couldn't send the message: region start_date is after end_date", true],
      ["Couldn't send the message: Failed to fetch", true],
    ]);
    // Errors never go back to the server as history.
    expect(bodies()[1].history).toEqual([{ role: 'user', text: 'one' }]);
  });

  it('sends nothing without text, files or a region', async () => {
    await sendMessage('   ');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(messages()).toEqual([]);
  });
});

describe('conversationHistory', () => {
  it('keeps the last 10 turns, each at most 2000 characters', () => {
    const store = useChatStore.getState();
    for (let i = 0; i < 12; i++) store.addMessage({ kind: 'user', text: `turn ${i}` });
    store.addMessage({ kind: 'assistant', text: 'x'.repeat(2500) });
    const history = conversationHistory(messages());
    expect(history).toHaveLength(10);
    expect(history[0].text).toBe('turn 3');
    expect(history[9]).toEqual({ role: 'assistant', text: 'x'.repeat(2000) });
  });
});

describe('useSendMessage', () => {
  it('invalidates the session after an analysis starts', async () => {
    fetchMock.mockResolvedValueOnce(sse([['analysis', { job_id: 'job-1', job: JOB }]]));
    const client = new QueryClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
    const { result } = renderHook(() => useSendMessage(), { wrapper });

    await act(() => result.current.send('Map open water with NDWI'));
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['session'] });
    expect(result.current.sending).toBe(false);
  });
});
