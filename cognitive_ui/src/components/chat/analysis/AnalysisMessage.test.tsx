import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../../../api/client';
import { SESSION_QUERY_KEY } from '../../../api/hooks/useSession';
import { useChatStore } from '../../../store/useChatStore';
import { SAMPLE } from '../../../test/fixtures';
import {
  EE_FAILED,
  MODEL_GATED,
  MODEL_GATED_MESSAGE,
  NDMI_CONFIRMED,
  NDMI_PROPOSAL,
  NDVI_CANCELLED,
  NDVI_DONE,
  NDVI_FAILED_AT_2,
  NDVI_FOLLOWUPS,
  PRITHVI_PREPARING,
} from '../../../test/jobEvents';
import NDVI_JOB from '../../../test/fixtures/jobs/ndvi-single.json';
import type { AnalysisMessage as AnalysisMessageData, JobEvent } from '../../../types';
import { AnalysisMessage } from './AnalysisMessage';

const JOB = 'job-1';
const PROMPT = 'Compare water before and after the dam breach';

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

/** An event stream that stays open until the request is aborted. */
function openStream(init?: RequestInit): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      init?.signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
    },
  });
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
}

let fetchMock: ReturnType<typeof vi.fn>;
let onSend: ReturnType<typeof vi.fn>;
let onOpenFilePicker: ReturnType<typeof vi.fn>;
let queryClient: QueryClient;
const calls = () => fetchMock.mock.calls.map(([url, init]) => `${(init as RequestInit | undefined)?.method ?? 'GET'} ${url as string}`);

function Live() {
  const message = useChatStore((s) => s.conversations[0]?.messages.find((m) => m.kind === 'analysis')) as AnalysisMessageData;
  return <AnalysisMessage message={message} prompt={PROMPT} onSend={onSend} onOpenFilePicker={onOpenFilePicker} />;
}

function renderAnalysis(events: JobEvent[], extra: Partial<AnalysisMessageData> = {}) {
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], pendingRegion: null, draft: '' });
  useChatStore.getState().addMessage({ kind: 'user', text: PROMPT });
  useChatStore.getState().addMessage({ kind: 'analysis', jobId: JOB, status: 'pending', text: '', events, ...extra });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/app']}>
        <Routes>
          <Route path="/app" element={<ol><Live /></ol>} />
          <Route path="/app/map" element={<p>Map page</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const message = () => screen.getByTestId('analysis-message');
const button = (name: string | RegExp) => screen.getByRole('button', { name });
const rows = () => screen.getAllByTestId('step-row').map((r) => [r.dataset.status, r.textContent]);

beforeEach(() => {
  localStorage.clear();
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  onSend = vi.fn();
  onOpenFilePicker = vi.fn();
  fetchMock = vi.fn(async (url: string, init?: RequestInit) => (url.includes('/events') ? openStream(init) : json(500, {})));
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('running', () => {
  it('intent, Running pill, one row per step with ✓ / spinner / queued, a progress bar and Cancel', async () => {
    // 9 s after step 2 started on the server clock.
    vi.spyOn(Date, 'now').mockReturnValue(((NDVI_DONE[5].data.t as number) + 9) * 1000);
    renderAnalysis(NDVI_DONE.slice(0, 6));
    expect(within(message()).getByText("I'll compute NDVI for 1 June from the sample.")).toBeTruthy();
    expect(screen.getByTestId('status-pill').textContent).toBe('Running');
    expect(screen.getByText('Step 2 of 3')).toBeTruthy();
    expect(rows()).toEqual([
      ['done', '✓Read hlsl_20230601.tif<1 s'],
      ['running', 'NDVI9 s'],
      ['queued', 'Interpretationqueued'],
    ]);
    expect(screen.getByRole('progressbar', { name: 'Analysis progress' }).getAttribute('aria-valuenow')).toBe('33');
    expect(button('Cancel').tagName).toBe('BUTTON');
    // It follows the job from its last event.
    await waitFor(() => expect(calls()).toContain(`GET /v1/jobs/${JOB}/events?after=6`));
  });

  it('Cancel posts to /cancel and shows the Cancelled pill', async () => {
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url.includes('/events')) return openStream(init);
      return json(200, { id: JOB, status: 'cancelled', events: NDVI_CANCELLED });
    });
    renderAnalysis(NDVI_DONE.slice(0, 6));
    fireEvent.click(button('Cancel'));
    await waitFor(() => expect(message().dataset.phase).toBe('cancelled'));
    expect(calls()).toContain(`POST /v1/jobs/${JOB}/cancel`);
    expect(screen.getByTestId('status-pill').textContent).toBe('–Cancelled');
    expect(screen.getByText('at step 2 of 3')).toBeTruthy();
    expect(rows().map(([s]) => s)).toEqual(['done', 'cancelled', 'skipped']);
    expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull();
  });

  it('Planning: the pill and the skeleton before any step starts', () => {
    renderAnalysis(NDVI_DONE.slice(0, 3));
    expect(screen.getByTestId('status-pill').textContent).toBe('Planning');
    expect(rows().map(([s]) => s)).toEqual(['queued', 'queued', 'queued']);
  });

  it('a Preparing model row shows its download: bar, bytes and time left', () => {
    renderAnalysis(PRITHVI_PREPARING);
    const prepare = screen.getAllByTestId('step-row')[1];
    expect(prepare.dataset.kind).toBe('prepare');
    expect(within(prepare).getByText('Preparing model · Prithvi EO v1 (100M)')).toBeTruthy();
    expect(within(prepare).getByText('Downloading Prithvi EO v1 (100M) · 450 MB · 50 MB done · 56 s left')).toBeTruthy();
    expect(within(prepare).getByRole('progressbar').getAttribute('aria-valuenow')).toBe('11');
  });
});

describe('proposal', () => {
  it('the proposal text with real Run · Change · Explain buttons', () => {
    renderAnalysis(NDMI_PROPOSAL);
    const proposal = screen.getByTestId('proposal');
    expect(proposal.textContent).toContain("I don't have a moisture-change algorithm.");
    expect(within(proposal).getByText('NDMI = (NIR − SWIR1)/(NIR + SWIR1)').tagName).toBe('CODE');
    for (const name of ['Run', 'Change', 'Explain']) expect(button(name).tagName).toBe('BUTTON');
    expect(screen.queryByTestId('analysis-timeline')).toBeNull();
  });

  it('Run confirms, then "Accepted · running below" and the composed timeline on the same message', async () => {
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url.includes('/events')) return openStream(init);
      if (url.endsWith('/confirm')) return json(202, { id: JOB, status: 'pending', events: NDMI_CONFIRMED.slice(0, 4) });
      return json(500, {});
    });
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
    renderAnalysis(NDMI_PROPOSAL);
    fireEvent.click(button('Run'));
    await screen.findByText('Accepted · running below');
    expect(calls()).toContain(`POST /v1/jobs/${JOB}/confirm`);
    // The quota is counted at confirm: the badge refreshes.
    expect(invalidate).toHaveBeenCalledWith({ queryKey: SESSION_QUERY_KEY });
    expect(screen.queryByRole('button', { name: 'Run' })).toBeNull();
    expect(screen.getByTestId('composed-badge').textContent).toBe('Composed by the assistant');
    expect(within(message()).getByText("I'll compute NDMI for 1 June and 9 June from the sample.")).toBeTruthy();
    expect(rows()).toHaveLength(4);
  });

  it.each([
    [409, { ok: false, error: { code: 'conflict', message: 'Job job-1 is not a proposal waiting for confirmation (it is running)' } }],
    [422, { ok: false, error: { code: 'validation_error', message: 'Step algorithms/band-math-change: Formula not allowed: `abs(B5)`.' } }],
    [429, { ok: false, error: { code: 'GUEST_QUOTA', message: 'Guests can run 5 analyses per hour on this server. The next one is available at 06:58 UTC.' } }],
  ])('a %s from confirm is shown under the buttons', async (status, body) => {
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => (url.includes('/events') ? openStream(init) : json(status, body)));
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
    renderAnalysis(NDMI_PROPOSAL);
    fireEvent.click(button('Run'));
    expect((await screen.findByRole('alert')).textContent).toBe(body.error.message);
    expect(button('Run')).toBeTruthy();
    if (status === 429) expect(invalidate).toHaveBeenCalledWith({ queryKey: SESSION_QUERY_KEY });
    else expect(invalidate).not.toHaveBeenCalled();
  });

  it('Change puts the prompt with the formula in the composer; Explain asks in the conversation', () => {
    renderAnalysis(NDMI_PROPOSAL);
    fireEvent.click(button('Change'));
    expect(useChatStore.getState().draft).toBe(`${PROMPT}, using NDMI = (B5 - B6) / (B5 + B6) and a change threshold of 0.2`);
    fireEvent.click(button('Explain'));
    expect(onSend).toHaveBeenCalledWith('Explain this proposal');
  });
});

describe('done', () => {
  it('collapses to one line with Show steps, the result card, the interpretation and follow-ups', async () => {
    fetchMock.mockImplementation(async (url: string) => (url.endsWith('/v1/jobs/job-1') ? json(200, NDVI_JOB) : json(500, {})));
    renderAnalysis(NDVI_DONE, { result: { summary: 'NDVI is high across the fields.' } });
    expect(screen.getByTestId('done-line').textContent).toBe('✓Done3 steps · <1 s · NDVIShow steps');
    expect(screen.queryByTestId('step-row')).toBeNull();
    expect(screen.getByTestId('interpretation').textContent).toBe('NDVI is high across the fields.');
    expect(screen.queryByTestId('composed-badge')).toBeNull();
    expect((await screen.findByTestId('result-card')).getAttribute('aria-label')).toBe('NDVI');
    // Finished: the record once for its result, no stream.
    expect(calls()).toEqual(['GET /v1/jobs/job-1']);
  });

  it('Show steps expands the rows and Hide steps folds them', () => {
    renderAnalysis(NDVI_DONE);
    fireEvent.click(button('Show steps'));
    expect(rows().map(([s]) => s)).toEqual(['done', 'done', 'done']);
    expect(button('Hide steps').getAttribute('aria-expanded')).toBe('true');
    fireEvent.click(button('Hide steps'));
    expect(screen.queryByTestId('step-row')).toBeNull();
  });

  it('each follow-up is a button that sends its full prompt', () => {
    renderAnalysis(NDVI_DONE);
    const chips = within(screen.getByRole('list', { name: 'Follow-ups' })).getAllByRole('button');
    expect(chips.map((c) => c.textContent)).toEqual(NDVI_FOLLOWUPS);
    fireEvent.click(chips[1]);
    expect(onSend).toHaveBeenCalledWith('Map open water with NDWI');
  });

  it('a composed analysis carries the badge on its Done line', () => {
    renderAnalysis(NDMI_CONFIRMED);
    expect(screen.getByText('Accepted · running below')).toBeTruthy();
    expect(within(screen.getByTestId('done-line')).getByTestId('composed-badge')).toBeTruthy();
  });
});

describe('failed', () => {
  it('stays open: the plain message, the code with show raw, and the fixes as buttons', () => {
    renderAnalysis(NDVI_FAILED_AT_2);
    expect(screen.getByTestId('status-pill').textContent).toBe('!Failed');
    expect(screen.getByText(/^at step 2 of 3 · /)).toBeTruthy();
    expect(rows().map(([s]) => s)).toEqual(['done', 'failed', 'skipped']);
    expect(screen.getByTestId('analysis-failure').textContent).toContain('Something went wrong while computing NDVI.');
    expect(screen.getByTestId('failure-code').textContent).toBe('INTERNAL');
    expect(screen.queryByTestId('failure-raw')).toBeNull();
    fireEvent.click(button('show raw'));
    expect(screen.getByTestId('failure-raw').textContent).toBe('Traceback: ValueError: boom');
    const fixes = within(screen.getByRole('group', { name: 'Fixes' })).getAllByRole('button');
    expect(fixes.map((b) => b.textContent)).toEqual(['Retry', 'Map open water with NDWI', 'Pick a region on the map']);
  });

  it('retry resends the same prompt with params; prompt sends its prompt', () => {
    renderAnalysis(NDVI_FAILED_AT_2);
    fireEvent.click(button('Retry'));
    expect(onSend).toHaveBeenCalledWith(PROMPT, { params: { mode: 'template' } });
    fireEvent.click(button('Map open water with NDWI'));
    expect(onSend).toHaveBeenLastCalledWith('Map open water with NDWI');
  });

  it('open_map goes to /app/map', () => {
    renderAnalysis(NDVI_FAILED_AT_2);
    fireEvent.click(button('Pick a region on the map'));
    expect(screen.getByText('Map page')).toBeTruthy();
  });

  it('upload opens the file picker; sample attaches the Kakhovka pair and resends the prompt', async () => {
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue(SAMPLE as never);
    renderAnalysis(EE_FAILED);
    expect(screen.getByText(/^while planning · \d+ s$/)).toBeTruthy();
    expect(screen.queryByTestId('step-row')).toBeNull();
    fireEvent.click(button('Use my own image'));
    expect(onOpenFilePicker).toHaveBeenCalledTimes(1);
    await act(async () => {
      fireEvent.click(button('Try the Kakhovka sample'));
    });
    expect(post).toHaveBeenCalledWith('/v1/sample');
    expect(useChatStore.getState().pendingAttachments.map((a) => a.path)).toEqual(SAMPLE.attachments.map((a) => a.path));
    expect(onSend).toHaveBeenCalledWith(PROMPT);
  });
});

describe('model gated for a guest (workspace-model-gated-for-guest.png)', () => {
  it('one assistant sentence with its fixes: no Failed pill, timeline, code or show raw', () => {
    renderAnalysis(MODEL_GATED);
    expect(message().dataset.phase).toBe('failed');
    expect(screen.getByTestId('analysis-refusal').textContent).toBe(MODEL_GATED_MESSAGE);
    expect(screen.queryByText("I'll run Prithvi MAE Reconstruction on the sample.")).toBeNull();
    expect(screen.queryByTestId('status-pill')).toBeNull();
    expect(screen.queryByTestId('analysis-timeline')).toBeNull();
    expect(screen.queryByTestId('failure-code')).toBeNull();
    expect(screen.queryByRole('button', { name: 'show raw' })).toBeNull();
    const fixes = within(screen.getByRole('group', { name: 'Fixes' })).getAllByRole('button');
    expect(fixes.map((b) => b.textContent)).toEqual(['See what I can do', 'Use a built-in analysis instead']);
    fireEvent.click(fixes[0]);
    expect(onSend).toHaveBeenCalledWith('What can you do?');
  });

  it('a model not installed (member) reads the same way', () => {
    renderAnalysis(
      MODEL_GATED.map((e) =>
        e.type === 'failed'
          ? { ...e, data: { ...e.data, code: 'MODEL_NOT_INSTALLED', message: "Prithvi EO v1 (100M) isn't installed on this server (450 MB) — ask an operator to install it." } }
          : e,
      ),
    );
    expect(screen.getByTestId('analysis-refusal').textContent).toMatch(/^Prithvi EO v1 \(100M\) isn't installed on this server/);
    expect(screen.queryByTestId('status-pill')).toBeNull();
  });
});

describe('status pills: glyph + word + hue', () => {
  it.each([
    [NDVI_DONE.slice(0, 1), 'starting', null],
    [NDVI_DONE.slice(0, 2), 'planning', 'Planning'],
    [NDVI_DONE.slice(0, 4), 'running', 'Running'],
    [NDVI_FAILED_AT_2, 'failed', '!Failed'],
    [NDVI_CANCELLED, 'cancelled', '–Cancelled'],
  ])('%#: %s', (events, phase, word) => {
    renderAnalysis(events);
    expect(message().dataset.phase).toBe(phase);
    const pill = screen.queryByTestId('status-pill');
    expect(pill?.textContent ?? null).toBe(word);
    if (pill) expect(pill.querySelector('[aria-hidden]')).toBeTruthy();
  });
});
