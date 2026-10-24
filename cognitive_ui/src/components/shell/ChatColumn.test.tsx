import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../../api/client';
import { sampleAttachments } from '../../api/sample';
import { useChatStore } from '../../store/useChatStore';
import { SAMPLE } from '../../test/fixtures';
import { ChatColumn } from './ChatColumn';

const HEALTH = { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: false, offline: true, provider: null, model: null } };

function renderColumn() {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/health') return HEALTH as never;
    throw new Error(`unexpected GET ${url}`);
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ChatColumn />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function sseReply(text: string): Response {
  return new Response(`event: reply_delta\ndata: ${JSON.stringify({ text })}\n\nevent: reply_done\ndata: ${JSON.stringify({ text })}\n\n`, {
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], pendingRegion: null, draft: '' });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const suggestionTexts = () =>
  within(screen.getByRole('list', { name: 'Suggestions' }))
    .getAllByRole('button')
    .map((b) => b.textContent);

describe('empty state', () => {
  it('without the sample: the three ways to add data and three suggestions', () => {
    renderColumn();
    expect(screen.getByRole('heading', { name: 'What do you want to know?' })).toBeTruthy();
    for (const name of ['+ Attach the Kakhovka sample', 'Upload GeoTIFF', 'Pick a region on the map']) {
      expect(screen.getByRole('button', { name }).tagName).toBe('BUTTON');
    }
    expect(suggestionTexts()).toEqual([
      'Compare water before and after the dam breach~50 s',
      'Detect field boundaries east of the river~2 min',
      'Classify land cover on the flooded fields~90 s',
    ]);
  });

  it('with the sample: its chip with ×, and five suggestions', () => {
    sampleAttachments(SAMPLE).forEach(useChatStore.getState().addAttachment);
    renderColumn();
    expect(screen.getByRole('heading', { name: 'What do you want to know about this area?' })).toBeTruthy();
    expect(screen.getByText(/^The Kakhovka sample is attached\./)).toBeTruthy();
    expect(screen.getAllByTestId('attachment-chip')).toHaveLength(1);
    expect(screen.queryByRole('button', { name: '+ Attach the Kakhovka sample' })).toBeNull();
    expect(suggestionTexts().slice(3)).toEqual([
      'Map moisture change with NDMI between the two datesComposed~60 s',
      'Fill the cloud gaps in the 9 June imagePrithvi · ~3 min',
    ]);

    fireEvent.click(screen.getByRole('button', { name: 'Remove kakhovka_hls_l30' }));
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
    expect(screen.getByRole('heading', { name: 'What do you want to know?' })).toBeTruthy();
  });

  it('a suggestion sends its prompt with the sample', async () => {
    const fetchMock = vi.fn(async () => sseReply('ok'));
    vi.stubGlobal('fetch', fetchMock);
    sampleAttachments(SAMPLE).forEach(useChatStore.getState().addAttachment);
    renderColumn();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^Compare water before and after the dam breach/ }));
    });
    const body = JSON.parse((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(body.message).toBe('Compare water before and after the dam breach');
    expect(body.attachments).toHaveLength(2);
    await screen.findByText('ok');
  });
});

describe('messages', () => {
  it('renders the user bubble with read-only chips, markdown replies with the avatar, and the analysis row', () => {
    const store = useChatStore.getState();
    store.addMessage({
      kind: 'user',
      text: 'Compare water before and after the dam breach',
      attachments: sampleAttachments(SAMPLE),
    });
    store.addMessage({ kind: 'assistant', text: 'I can:\n\n- map **water**\n- map vegetation' });
    store.addMessage({
      kind: 'analysis',
      jobId: 'job-1',
      status: 'pending',
      text: '',
      events: [{ seq: 1, type: 'intent', data: { text: "I'll compute NDWI for 1 June and 9 June from the sample." } }],
    });
    store.addMessage({ kind: 'assistant', text: 'Guests can run 5 analyses per hour.', error: true });
    renderColumn();

    const user = screen.getByTestId('user-message');
    expect(within(user).getByTestId('attachment-chip').textContent).toBe(
      'kakhovka_hls_l30HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09',
    );
    expect(within(user).queryByRole('button')).toBeNull();

    const [reply, error] = screen.getAllByTestId('assistant-message');
    expect(reply.textContent).toContain('DT');
    expect(within(reply).getByText('water').tagName).toBe('STRONG');
    expect(within(reply).getAllByRole('listitem')).toHaveLength(2);
    expect(within(error).getByRole('alert').textContent).toBe('Guests can run 5 analyses per hour.');

    expect(screen.getByTestId('analysis-message').textContent).toContain("I'll compute NDWI for 1 June and 9 June");
    expect(screen.getByLabelText('Message').getAttribute('placeholder')).toBe('Ask a follow-up…');
  });

  it('Enter in the composer sends the draft and streams the reply', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => sseReply('**Hello!** I map water.')));
    renderColumn();
    const input = screen.getByLabelText('Message');
    fireEvent.change(input, { target: { value: 'hello' } });
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' });
    });
    await waitFor(() => expect(screen.getByText('Hello!').tagName).toBe('STRONG'));
    expect(screen.getByTestId('user-message').textContent).toBe('hello');
    expect((input as HTMLTextAreaElement).value).toBe('');
  });
});
