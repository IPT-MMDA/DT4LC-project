import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../../api/client';
import { AppRoutes } from '../../routes';
import { useAppStore } from '../../store/useAppStore';
import { useChatStore } from '../../store/useChatStore';
import { GUEST_SESSION, OPERATOR_SESSION, SAMPLE } from '../../test/fixtures';
import type { SessionInfo } from '../../types';

// jsdom has no WebGL: the map panes render a stand-in.
vi.mock('../map/MapContainer', () => ({ MapContainer: () => <div data-testid="maplibre-map" /> }));
vi.mock('../../pages/MapPage', () => ({ MapPage: () => <div data-testid="map-page" /> }));

const HEALTH = { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: false, offline: true, provider: null, model: null } };

function mockApi(session: SessionInfo = GUEST_SESSION) {
  const get = vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/session') return session as never;
    if (url === '/v1/health') return HEALTH as never;
    throw new Error(`unexpected GET ${url}`);
  });
  const post = vi.spyOn(apiClient, 'post').mockImplementation(async (url: string) => {
    if (url === '/v1/sample') return SAMPLE as never;
    throw new Error(`unexpected POST ${url}`);
  });
  return { get, post };
}

// matchMedia stand-in: `wide` decides whether the ≥1440 px query matches.
function setViewport(wide: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: wide && query.includes('min-width: 1440px'),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

const location = { current: '' };
function LocationProbe() {
  const loc = useLocation();
  location.current = loc.pathname + loc.search;
  return null;
}

function renderApp(url: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[url]}>
        <AppRoutes />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], draft: '' });
  useAppStore.setState({ theme: 'light' });
  setViewport(true);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('Workspace shell', () => {
  it('renders the rail, the conversation list and the chat column', async () => {
    mockApi();
    renderApp('/app');

    const rail = screen.getByRole('navigation', { name: 'Main' });
    for (const [name, href] of [
      ['Chat', '/app'],
      ['Map', '/app/map'],
      ['Library', '/app/library'],
      ['Can do', '/app/capabilities'],
    ]) {
      expect(within(rail).getByRole('link', { name }).getAttribute('href')).toBe(href);
    }
    expect(within(rail).getByRole('link', { name: 'Chat' }).getAttribute('aria-current')).toBe('page');
    expect(within(rail).getByRole('link', { name: 'DT4LC home' })).toBeTruthy();
    expect(within(rail).getByRole('button', { name: 'Switch to dark theme' })).toBeTruthy();
    expect(within(rail).getByRole('button', { name: 'Account: Guest' })).toBeTruthy();

    const list = screen.getByRole('complementary', { name: 'Conversations' });
    expect(within(list).getByRole('button', { name: '+ New' })).toBeTruthy();
    expect(within(list).getByText(/No conversations yet/)).toBeTruthy();

    const chat = screen.getByRole('main', { name: 'Chat' });
    expect(within(chat).getByRole('heading', { name: 'What do you want to know?' })).toBeTruthy();
    expect(within(chat).getByLabelText('Message')).toBeTruthy();
  });

  it('shows the guest badge from GET /v1/session', async () => {
    mockApi();
    renderApp('/app');
    const badge = await screen.findByTestId('session-badge');
    expect(badge.textContent).toBe('Guest. 4 of 5 analyses left this hour · kept 24 h · no account.');
  });

  it('lists conversations with their meta, highlights the active one and shows it in the header', async () => {
    mockApi();
    const store = useChatStore.getState();
    store.addMessage({ kind: 'user', text: 'Older chat' });
    store.newConversation();
    store.addMessage({
      kind: 'user',
      text: 'Kakhovka reservoir — water change after breach',
      attachments: [{ id: 's', filename: 'a.tif', path: '/u/a.tif', group: { name: 'kakhovka_hls_l30', line: '' } }],
    });
    store.addMessage({ kind: 'analysis', jobId: 'job-1', status: 'done', text: '', events: [], result: { card: 'change' } });
    renderApp('/app');

    const list = screen.getByRole('complementary', { name: 'Conversations' });
    const items = within(list).getAllByTestId('conversation').map((li) => li.querySelector('button') as HTMLButtonElement);
    expect(items.map((b) => b.textContent)).toEqual([
      'Kakhovka reservoir — water change after breach1 file · 1 result',
      'Older chat',
    ]);
    expect(items[0].getAttribute('aria-current')).toBe('true');
    expect(items[1].getAttribute('aria-current')).toBeNull();

    const header = within(screen.getByRole('main', { name: 'Chat' })).getByRole('banner');
    expect(header.textContent).toBe('Kakhovka reservoir — water change after breach1 file · 1 result');

    fireEvent.click(items[1]);
    expect(within(screen.getByRole('main', { name: 'Chat' })).getByRole('heading', { level: 1 }).textContent).toBe('Older chat');

    fireEvent.click(within(list).getByRole('button', { name: '+ New' }));
    expect(useChatStore.getState().activeConversationId).toBeNull();
    expect(screen.getByRole('heading', { name: 'What do you want to know?' })).toBeTruthy();
  });
});

describe('docked map', () => {
  it('renders beside the chat at ≥1440 px', () => {
    mockApi();
    renderApp('/app');
    const pane = screen.getByRole('region', { name: 'Map' });
    expect(within(pane).getByTestId('maplibre-map')).toBeTruthy();
    expect(within(pane).getByRole('button', { name: 'Fit to result' })).toBeTruthy();
    expect(within(pane).getByRole('button', { name: 'Compare' })).toBeTruthy();
    expect(within(pane).getByRole('link', { name: 'Open full map ↗' }).getAttribute('href')).toBe('/app/map');
    expect(within(pane).getByRole('region', { name: 'Layers' }).textContent).toContain('opacity');
  });

  it('is not rendered below 1440 px; the Map rail item opens /app/map', async () => {
    setViewport(false);
    mockApi();
    renderApp('/app');
    expect(screen.queryByRole('region', { name: 'Map' })).toBeNull();
    expect(screen.queryByTestId('maplibre-map')).toBeNull();

    fireEvent.click(screen.getByRole('link', { name: 'Map' }));
    expect(location.current).toBe('/app/map');
    expect(screen.getByTestId('map-page')).toBeTruthy();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
  });
});

describe('routes', () => {
  // `/` without a query is the guest landing (pages/Landing.test.tsx).
  it.each([
    ['/?sample=0', '/app?sample=0'],
    ['/chat', '/app'],
    ['/dashboard', '/app'],
    ['/map', '/app/map'],
    ['/jobs', '/app/library'],
    ['/jobs/abc123', '/app/results/abc123'],
    ['/data', '/app/library'],
    ['/models', '/app/capabilities'],
    ['/nowhere', '/app'],
  ])('%s redirects to %s', async (from, to) => {
    mockApi();
    renderApp(from);
    await waitFor(() => expect(location.current).toBe(to));
  });

  it.each([
    ['/app/library', 'Library'],
  ])('%s is a placeholder inside the shell', (url, title) => {
    mockApi();
    renderApp(url);
    expect(screen.getByRole('heading', { name: title })).toBeTruthy();
    expect(screen.getByText('Arrives in the next run.')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Back to chat' }).getAttribute('href')).toBe('/app');
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
  });

  it('/app/capabilities is the Capabilities page inside the shell', () => {
    mockApi();
    renderApp('/app/capabilities');
    expect(screen.getByRole('main', { name: 'What DT4LC can do · on this server' })).toBeTruthy();
    expect(screen.queryByText('Arrives in the next run.')).toBeNull();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
  });

  it('/app/results/:id opens the conversation that holds the job', async () => {
    mockApi();
    const { conversationId } = useChatStore.getState().addMessage({ kind: 'user', text: 'NDVI please' });
    useChatStore.getState().addMessage({ kind: 'analysis', jobId: 'job-42', status: 'done', text: '', events: [] });
    useChatStore.getState().newConversation();

    renderApp('/app/results/job-42');
    await waitFor(() => expect(location.current).toBe('/app'));
    expect(useChatStore.getState().activeConversationId).toBe(conversationId);
  });

  it('/app/results/:id of an unknown job is the placeholder', () => {
    mockApi();
    renderApp('/app/results/elsewhere');
    expect(screen.getByRole('heading', { name: 'Result' })).toBeTruthy();
    expect(location.current).toBe('/app/results/elsewhere');
  });

  it('/app/settings is for operators only', async () => {
    mockApi(GUEST_SESSION);
    renderApp('/settings');
    await waitFor(() => expect(location.current).toBe('/app'));
    cleanup();
    vi.restoreAllMocks();

    mockApi(OPERATOR_SESSION);
    renderApp('/app/settings');
    expect(await screen.findByRole('heading', { name: 'Settings' })).toBeTruthy();
  });
});

describe('account menu', () => {
  it('shows the role from GET /v1/session and toggles the theme', async () => {
    mockApi();
    renderApp('/app');
    fireEvent.click(screen.getByRole('button', { name: 'Account: Guest' }));

    const menu = screen.getByRole('menu', { name: 'Account' });
    expect((await within(menu).findByTestId('role-badge')).textContent).toBe('Guest');
    expect(within(menu).queryByRole('menuitem', { name: 'Settings' })).toBeNull();
    const signIn = within(menu).getByRole('menuitem', { name: /Sign in/ });
    expect((signIn as HTMLButtonElement).disabled).toBe(true);
    expect(signIn.textContent).toContain('Coming soon');

    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Dark theme' }));
    expect(useAppStore.getState().theme).toBe('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');

    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Light theme' }));
    expect(document.documentElement.dataset.theme).toBe('light');

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('offers Settings to operators', async () => {
    mockApi(OPERATOR_SESSION);
    renderApp('/app');
    fireEvent.click(await screen.findByRole('button', { name: 'Account: Operator' }));
    const menu = screen.getByRole('menu', { name: 'Account' });
    expect(within(menu).getByTestId('role-badge').textContent).toBe('Operator');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Settings' }));
    expect(location.current).toBe('/app/settings');
  });

  it('the rail theme button toggles the theme too', () => {
    mockApi();
    renderApp('/app');
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark theme' }));
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(screen.getByRole('button', { name: 'Switch to light theme' })).toBeTruthy();
  });
});

describe('?sample=1', () => {
  it('calls POST /v1/sample and puts the pair in the composer as one chip', async () => {
    const { post } = mockApi();
    renderApp('/?sample=1');

    const chip = await screen.findByTestId('attachment-chip');
    expect(post).toHaveBeenCalledWith('/v1/sample');
    expect(chip.textContent).toBe('kakhovka_hls_l30HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09×');
    expect(within(chip).getByRole('button', { name: 'Remove kakhovka_hls_l30' })).toBeTruthy();
    expect(screen.getAllByTestId('attachment-chip')).toHaveLength(1);
    expect(useChatStore.getState().pendingAttachments.map((a) => a.filename)).toEqual([
      'hlsl_20230601.tif',
      'hlsl_20230609.tif',
    ]);
    await waitFor(() => expect(location.current).toBe('/app'));
  });

  it('?sample=0 leaves the composer empty', async () => {
    const { post } = mockApi();
    renderApp('/app?sample=0');
    await act(async () => {});
    expect(post).not.toHaveBeenCalled();
    expect(screen.queryByTestId('attachment-chip')).toBeNull();
  });
});
