import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../../api/client';
import { sampleAttachments } from '../../api/sample';
import { useAppStore, type DrawnRegion } from '../../store/useAppStore';
import { useChatStore } from '../../store/useChatStore';
import { GUEST_SESSION, SAMPLE } from '../../test/fixtures';
import type { SessionInfo } from '../../types';
import { Composer } from './Composer';

const OFFLINE = 'Assistant offline — analyses still run; answers are limited to the catalog.';

function health(offline: boolean, session?: SessionInfo) {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/health') {
      return { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: !offline, offline, provider: null, model: null } } as never;
    }
    if (url === '/v1/session' && session) return session as never;
    throw new Error(`unexpected GET ${url}`);
  });
}

function renderComposer(props: Partial<Parameters<typeof Composer>[0]> = {}) {
  const onSend = vi.fn();
  const onOpenFilePicker = vi.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/app']}>
        <Routes>
          <Route
            path="/app"
            element={<Composer hasMessages={false} sending={false} onSend={onSend} onOpenFilePicker={onOpenFilePicker} {...props} />}
          />
          <Route path="/app/map" element={<p>map page</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { onSend, onOpenFilePicker, input: screen.getByLabelText('Message') as HTMLTextAreaElement };
}

const DRAWN: DrawnRegion = {
  id: 'region-1',
  name: 'Dnipro floodplain · Kherson',
  type: 'polygon',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[32.5, 46.5], [32.6, 46.5], [32.6, 46.6], [32.5, 46.6], [32.5, 46.5]]] },
  },
  bbox: [32.5, 46.5, 32.6, 46.6],
  createdAt: '2026-10-23T00:00:00Z',
};

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], pendingRegion: null, draft: '' });
  useAppStore.setState({ selectedRegion: null, drawnRegions: [] });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('Composer keyboard', () => {
  it('Enter sends; Shift+Enter does not (the textarea inserts the newline)', () => {
    health(false);
    const { onSend, input } = renderComposer();
    fireEvent.change(input, { target: { value: 'Map open water' } });

    const shiftEnter = fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(shiftEnter).toBe(true);  // not prevented: the browser inserts the newline
    expect(onSend).not.toHaveBeenCalled();

    const enter = fireEvent.keyDown(input, { key: 'Enter' });
    expect(enter).toBe(false);  // prevented: no newline
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it('Enter does nothing while the draft is empty or a message is in flight', () => {
    health(false);
    const { onSend, input } = renderComposer();
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSend).not.toHaveBeenCalled();
    cleanup();

    useChatStore.getState().setDraft('hello');
    const again = renderComposer({ sending: true });
    fireEvent.keyDown(again.input, { key: 'Enter' });
    expect(again.onSend).not.toHaveBeenCalled();
    expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe('Composer controls', () => {
  it('every control is a real <button>', () => {
    health(false);
    useChatStore.getState().setDraft('x');
    const { onSend, onOpenFilePicker } = renderComposer();
    for (const name of ['Attach GeoTIFF', 'From map', 'From library', 'Send']) {
      const control = screen.getByRole('button', { name });
      expect(control.tagName).toBe('BUTTON');
      expect(control.getAttribute('type')).toBe('button');
    }
    fireEvent.click(screen.getByRole('button', { name: 'Attach GeoTIFF' }));
    expect(onOpenFilePicker).toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));
    expect(onSend).toHaveBeenCalled();
    expect(screen.getByText('Enter to send · Shift+Enter newline')).toBeTruthy();
  });

  it('shows the empty placeholder, then "Ask a follow-up…" once the conversation has messages', () => {
    health(false);
    expect(renderComposer().input.placeholder).toBe('Describe an analysis… or drop GeoTIFFs here');
    cleanup();
    expect(renderComposer({ hasMessages: true }).input.placeholder).toBe('Ask a follow-up…');
  });

  it('shows file chips with a remove × once the conversation has messages', () => {
    health(false);
    sampleAttachments(SAMPLE).forEach(useChatStore.getState().addAttachment);
    renderComposer({ hasMessages: true });
    const chip = screen.getByTestId('attachment-chip');
    expect(chip.textContent).toContain('kakhovka_hls_l30');
    fireEvent.click(within(chip).getByRole('button', { name: 'Remove kakhovka_hls_l30' }));
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
  });
});

describe('the assistant offline line', () => {
  it('appears above the controls when GET /v1/health says the assistant is offline', async () => {
    health(true);
    renderComposer();
    expect((await screen.findByTestId('assistant-offline')).textContent).toBe(OFFLINE);
  });

  it('is absent when the assistant is connected', async () => {
    health(false);
    renderComposer();
    await screen.findByLabelText('Message');
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByText(OFFLINE)).toBeNull();
  });
});

describe('the guest quota line', () => {
  const QUOTA = "You've used your 5 analyses for this hour — the next one is available at 14:05.";
  const guest = (remaining: number): SessionInfo => ({
    ...GUEST_SESSION,
    analyses: { limit: 5, used: 5 - remaining, remaining, resets_at: new Date(2026, 9, 23, 14, 5).toISOString() },
  });

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date(2026, 9, 23, 13, 40));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('at 0 left: one line with the local reset time; typing and sending still work', async () => {
    health(true, guest(0));
    const { onSend, input } = renderComposer();
    expect((await screen.findByTestId('quota-line')).textContent).toBe(QUOTA);
    expect(screen.getByTestId('assistant-offline')).toBeTruthy();
    expect(input.disabled).toBe(false);
    fireEvent.change(input, { target: { value: 'What is NDWI?' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it('at 1 left: no line', async () => {
    health(false, guest(1));
    renderComposer();
    await screen.findByLabelText('Message');
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByTestId('quota-line')).toBeNull();
  });
});

describe('From map', () => {
  it("attaches the map's selected region with the defaults, as a removable region chip", () => {
    health(false);
    useAppStore.setState({ drawnRegions: [DRAWN], selectedRegion: DRAWN });
    sampleAttachments(SAMPLE).forEach(useChatStore.getState().addAttachment);
    renderComposer({ hasMessages: true });

    fireEvent.click(screen.getByRole('button', { name: 'From map' }));
    const region = useChatStore.getState().pendingRegion;
    // The sample is in the composer, so its dates apply.
    expect(region).toMatchObject({ id: 'region-1', name: 'Dnipro floodplain · Kherson', dataset: 'HLS L30', cloud_max: 20, start_date: '2023-06-01', end_date: '2023-06-09' });
    const chip = screen.getByTestId('region-chip');
    expect(chip.textContent).toMatch(/^Region · Dnipro floodplain · Kherson8\d km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud×$/);

    fireEvent.click(within(chip).getByRole('button', { name: 'Remove region' }));
    expect(useChatStore.getState().pendingRegion).toBeNull();
  });

  it('opens the map when no region is selected', () => {
    health(false);
    renderComposer();
    fireEvent.click(screen.getByRole('button', { name: 'From map' }));
    expect(screen.getByText('map page')).toBeTruthy();
    expect(useChatStore.getState().pendingRegion).toBeNull();
  });
});

describe('From library', () => {
  it("lists the caller's uploads from GET /v1/files and attaches the one picked", async () => {
    health(false);
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          ok: true,
          count: 2,
          files: [
            { id: 'f1', filename: 'forest_2024.tif', path: '/u/f1_forest_2024.tif', size: [10, 10], crs: null, bounds: [0, 0, 1, 1], size_bytes: 7, source: 'upload', modified: 1, metadata: { line: 'Sentinel-2 · B2 B3 B4 B8 · 2024-07-11' } },
            { id: 'e1', filename: 'export.tif', path: '/x/export.tif', size: [10, 10], crs: null, bounds: [0, 0, 1, 1], size_bytes: 7, source: 'export', modified: 1 },
          ],
        }),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);
    renderComposer();

    fireEvent.click(screen.getByRole('button', { name: 'From library' }));
    const picker = screen.getByRole('dialog', { name: 'Your uploads' });
    const pick = await within(picker).findByRole('button', { name: /forest_2024\.tif/ });
    expect(within(picker).queryByText('export.tif')).toBeNull();
    fireEvent.click(pick);

    expect(useChatStore.getState().pendingAttachments).toMatchObject([
      { id: 'f1', filename: 'forest_2024.tif', path: '/u/f1_forest_2024.tif', mime_type: 'image/tiff' },
    ]);
    expect(screen.queryByRole('dialog')).toBeNull();
    vi.unstubAllGlobals();
  });
});
