import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useChatStore } from '../store/useChatStore';
import { NDVI_DONE } from '../test/jobEvents';
import type { AnalysisMessage, JobEvent } from '../types';
import { activeJobStreams, interpretationOf, subscribeJobEvents } from './useJobEvents';

const JOB = 'job-1';
const SUMMARY = 'NDVI is high across the floodplain fields: a mean of 0.55.';

/** An SSE body of `events`; `open` keeps it from ending until the request is aborted. */
function sse(events: JobEvent[], { open = false, signal }: { open?: boolean; signal?: AbortSignal } = {}): Response {
  const text = events.map((e) => `id: ${e.seq}\nevent: ${e.type}\ndata: ${JSON.stringify(e.data)}\n\n`).join('');
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      if (!open) controller.close();
      signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
    },
  });
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
}

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const jobRecord = (events: JobEvent[]) => ({
  id: JOB,
  status: 'completed',
  events,
  result: { execution: { artifacts: { agent_result: { summary: SUMMARY } } } },
});

let fetchMock: ReturnType<typeof vi.fn>;
const urls = () => fetchMock.mock.calls.map(([url]) => url as string);
const message = () => useChatStore.getState().conversations[0]?.messages.find((m) => m.kind === 'analysis') as AnalysisMessage;
const until = (check: () => boolean) => vi.waitFor(() => expect(check()).toBe(true), { timeout: 3000, interval: 10 });

function seed(events: JobEvent[]) {
  useChatStore.setState({ conversations: [], activeConversationId: null });
  useChatStore.getState().addMessage({ kind: 'user', text: 'calculate ndvi' });
  useChatStore.getState().addMessage({ kind: 'analysis', jobId: JOB, status: 'pending', text: '', events });
}

beforeEach(() => {
  localStorage.clear();
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('subscribeJobEvents', () => {
  it('streams the events into the message, closes on done and reads the interpretation', async () => {
    seed(NDVI_DONE.slice(0, 1));
    fetchMock.mockImplementation(async (url: string) =>
      url.endsWith('/events?after=1') ? sse(NDVI_DONE.slice(1)) : json(200, jobRecord(NDVI_DONE)),
    );

    const release = subscribeJobEvents(JOB);
    await until(() => message().result?.summary === SUMMARY);
    release();

    expect(urls()).toEqual([`/v1/jobs/${JOB}/events?after=1`, `/v1/jobs/${JOB}`]);
    expect(message().events.map((e) => (e as JobEvent).seq)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(message().status).toBe('done');
    expect(activeJobStreams()).toEqual([]);
  });

  it('reconnects with ?after=<last seq> when the stream ends before a terminal event', async () => {
    seed([]);
    let streams = 0;
    fetchMock.mockImplementation(async (url: string) => {
      if (!url.includes('/events')) return json(200, jobRecord(NDVI_DONE));
      streams += 1;
      return streams === 1 ? sse(NDVI_DONE.slice(0, 5)) : sse(NDVI_DONE.slice(5));
    });

    const release = subscribeJobEvents(JOB);
    await until(() => message().result?.summary === SUMMARY);
    release();

    expect(urls()).toEqual([`/v1/jobs/${JOB}/events`, `/v1/jobs/${JOB}/events?after=5`, `/v1/jobs/${JOB}`]);
    expect(message().events).toHaveLength(10);
  });

  it('a network error: the job record fills the gap (the reload source), then it reconnects if still running', async () => {
    seed(NDVI_DONE.slice(0, 3));
    let streams = 0;
    fetchMock.mockImplementation(async (url: string) => {
      if (!url.includes('/events')) return json(200, { ...jobRecord(NDVI_DONE.slice(0, 6)), status: 'running', result: null });
      streams += 1;
      if (streams === 1) throw new TypeError('Failed to fetch');
      return sse(NDVI_DONE.slice(6));
    });

    const release = subscribeJobEvents(JOB);
    await until(() => message().status === 'done');
    release();

    expect(urls().slice(0, 3)).toEqual([`/v1/jobs/${JOB}/events?after=3`, `/v1/jobs/${JOB}`, `/v1/jobs/${JOB}/events?after=6`]);
    expect(message().events).toHaveLength(10);
  });

  it('one stream per job however many follow it; the last to leave closes it', async () => {
    seed(NDVI_DONE.slice(0, 3));
    const signals: AbortSignal[] = [];
    fetchMock.mockImplementation(async (_url: string, init: RequestInit) => {
      signals.push(init.signal as AbortSignal);
      return sse(NDVI_DONE.slice(3, 4), { open: true, signal: init.signal as AbortSignal });
    });

    const a = subscribeJobEvents(JOB);
    const b = subscribeJobEvents(JOB);
    await until(() => message().events.length === 4);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(activeJobStreams()).toEqual([JOB]);

    a();
    expect(signals[0].aborted).toBe(false);
    b();
    expect(signals[0].aborted).toBe(true);
    expect(activeJobStreams()).toEqual([]);

    // Coming back (the conversation is selected again) resumes after the last event.
    const c = subscribeJobEvents(JOB);
    await until(() => fetchMock.mock.calls.length === 2);
    expect(urls()[1]).toBe(`/v1/jobs/${JOB}/events?after=4`);
    c();
  });

  it('a finished analysis opens no stream', async () => {
    seed(NDVI_DONE);
    const release = subscribeJobEvents(JOB);
    await new Promise((r) => setTimeout(r, 20));
    release();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('404: the job is gone; says so and does not retry', async () => {
    seed(NDVI_DONE.slice(0, 2));
    fetchMock.mockResolvedValue(json(404, { ok: false, error: { code: 'not_found', message: 'Job job-1 not found' } }));
    const release = subscribeJobEvents(JOB);
    await until(() => Boolean(message().error));
    await new Promise((r) => setTimeout(r, 600));
    release();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(message().error).toBe('This analysis is no longer on the server.');
  });
});

describe('interpretationOf', () => {
  it('the agent-analysis summary', () => {
    expect(interpretationOf(jobRecord(NDVI_DONE))).toBe(SUMMARY);
  });
  it('none when the assistant was offline, or no result', () => {
    const offline = { result: { execution: { artifacts: { agent_result: { summary: 'Analysis complete. Results available but summarization failed: No available LLM providers' } } } } };
    expect(interpretationOf(offline)).toBeUndefined();
    expect(interpretationOf({ result: null })).toBeUndefined();
  });
});
