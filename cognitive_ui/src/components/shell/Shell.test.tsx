import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../../api/client';
import { AppRoutes } from '../../routes';
import { useAppStore } from '../../store/useAppStore';
import { useChatStore } from '../../store/useChatStore';
import { GUEST_SESSION, MEMBER_SESSION, OPERATOR_SESSION, SAMPLE } from '../../test/fixtures';
import SETTINGS from '../../test/fixtures/settings/operator.json';
import type { SessionInfo } from '../../types';

// jsdom has no WebGL: the map panes render a stand-in.
vi.mock('../map/MapContainer', () => ({ MapContainer: () => <div data-testid="maplibre-map" /> }));
vi.mock('../../pages/MapPage', () => ({ MapPage: () => <div data-testid="map-page" /> }));

const HEALTH = { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: false, offline: true, provider: null, model: null } };

function mockApi(session: SessionInfo = GUEST_SESSION) {
  const get = vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/session') return session as never;
    if (url === '/v1/health') return HEALTH as never;
    if (url === '/v1/settings') return SETTINGS as never;
    throw new Error(`unexpected GET ${url}`);
  });
  const post = vi.spyOn(apiClient, 'post').mockImplementation(async (url: string) => {
    if (url === '/v1/sample') return SAMPLE as never;
    if (url === '/v1/auth/logout') return GUEST_SESSION as never;
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

function renderApp(url: string, client = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
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

  it('/app/library is the Library page inside the shell', () => {
    mockApi();
    renderApp('/app/library');
    expect(screen.getByRole('heading', { name: 'Library' })).toBeTruthy();
    expect(screen.getByTestId('library-page')).toBeTruthy();
    expect(screen.queryByText('Arrives in the next run.')).toBeNull();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
  });

  it('/app/capabilities is the Capabilities page inside the shell', () => {
    mockApi();
    renderApp('/app/capabilities');
    expect(screen.getByRole('main', { name: 'What DT4LC can do · on this server' })).toBeTruthy();
    expect(screen.queryByText('Arrives in the next run.')).toBeNull();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
  });

  it('/app/results/:id is Result detail inside the shell, under the Library in the rail', async () => {
    mockApi();
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ ok: false, error: { code: 'not_found', message: 'Job elsewhere not found' } }), { status: 404 })));
    renderApp('/app/results/elsewhere');
    expect(await screen.findByTestId('result-unavailable')).toBeTruthy();
    expect(location.current).toBe('/app/results/elsewhere');
    const rail = screen.getByRole('navigation', { name: 'Main' });
    expect(within(rail).getByRole('link', { name: 'Library' }).getAttribute('aria-current')).toBe('page');
    expect(within(rail).getByRole('link', { name: 'Chat' }).getAttribute('aria-current')).toBeNull();
  });

  it('/app/settings is the Settings page for operators', async () => {
    mockApi(OPERATOR_SESSION);
    renderApp('/settings');
    await waitFor(() => expect(location.current).toBe('/app/settings'));
    expect(await screen.findByRole('heading', { name: 'Settings · this server' })).toBeTruthy();
    expect(screen.queryByTestId('not-found')).toBeNull();
  });

  it.each([
    ['guest', GUEST_SESSION],
    ['member', MEMBER_SESSION],
  ])('/app/settings is the not-found page for a %s, as an unknown /app path', async (_, session) => {
    const { get } = mockApi(session);
    renderApp('/app/settings');
    expect(await screen.findByTestId('not-found')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Page not found' })).toBeTruthy();
    expect(location.current).toBe('/app/settings');
    expect(screen.queryByRole('heading', { name: 'Settings · this server' })).toBeNull();
    expect(get.mock.calls.some(([url]) => url === '/v1/settings')).toBe(false);
    cleanup();

    renderApp('/app/no-such-page');
    expect(await screen.findByTestId('not-found')).toBeTruthy();
    expect(screen.getByText('/app/no-such-page')).toBeTruthy();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
  });
});

describe('account menu', () => {
  const openMenu = async (name: string) => {
    fireEvent.click(await screen.findByRole('button', { name }));
    return screen.getByRole('menu', { name: 'Account' });
  };

  it('guest without sign-in on the server: the role, the theme, and no sign-in button', async () => {
    mockApi();
    renderApp('/app');
    const menu = await openMenu('Account: Guest');

    expect((await within(menu).findByTestId('role-badge')).textContent).toBe('Guest');
    expect(within(menu).queryByRole('menuitem', { name: 'Settings' })).toBeNull();
    expect(within(menu).getByTestId('sign-in-unavailable').textContent).toBe('Sign-in not configured on this server');
    expect(within(menu).queryByRole('menuitem', { name: /Sign in/ })).toBeNull();
    expect(within(menu).queryByRole('button', { name: /Sign in/ })).toBeNull();
    expect(within(menu).queryByRole('menuitem', { name: 'Sign out' })).toBeNull();

    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Dark theme' }));
    expect(useAppStore.getState().theme).toBe('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');

    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Light theme' }));
    expect(document.documentElement.dataset.theme).toBe('light');

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('guest with sign-in configured: a real link to the GitHub login route', async () => {
    mockApi({ ...GUEST_SESSION, sign_in: { configured: true } });
    renderApp('/app');
    const menu = await openMenu('Account: Guest');
    const link = await within(menu).findByRole('menuitem', { name: 'Sign in with GitHub' });
    expect(link.tagName).toBe('A');
    expect(link.getAttribute('href')).toBe('/v1/auth/github/login');
    expect(within(menu).queryByTestId('sign-in-unavailable')).toBeNull();
  });

  it('member: name, login, email and the Member badge; sign out, no Settings', async () => {
    mockApi(MEMBER_SESSION);
    renderApp('/app');
    const button = await screen.findByRole('button', { name: 'Account: Mira Member (Member)' });
    expect(button.getAttribute('title')).toBe('Mira Member');
    expect(button.textContent).toBe('M');
    const menu = await openMenu('Account: Mira Member (Member)');
    const user = within(menu).getByTestId('account-user');
    expect(user.textContent).toContain('Mira Member');
    expect(user.textContent).toContain('@mira');
    expect(user.textContent).toContain('mira@unige.ch');
    expect(within(menu).getByTestId('role-badge').textContent).toBe('Member');
    expect(within(menu).getByRole('menuitem', { name: 'Sign out' })).toBeTruthy();
    expect(within(menu).queryByRole('menuitem', { name: 'Settings' })).toBeNull();
    expect(within(menu).queryByRole('menuitem', { name: /Sign in/ })).toBeNull();
    expect(within(menu).queryByTestId('sign-in-unavailable')).toBeNull();
  });

  it('a user without a name: the login stands in', async () => {
    mockApi({ ...MEMBER_SESSION, user: { ...MEMBER_SESSION.user!, name: null } });
    renderApp('/app');
    expect(await screen.findByRole('button', { name: 'Account: mira (Member)' })).toBeTruthy();
  });

  it('operator: Settings opens /app/settings', async () => {
    mockApi(OPERATOR_SESSION);
    renderApp('/app');
    const menu = await openMenu('Account: Oscar Operator (Operator)');
    expect(within(menu).getByTestId('role-badge').textContent).toBe('Operator');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Settings' }));
    expect(location.current).toBe('/app/settings');
  });

  it('Sign out: POST /v1/auth/logout, refetch session, capabilities, jobs and files, then go to /', async () => {
    const { get, post } = mockApi(MEMBER_SESSION);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    renderApp('/app/library', client);
    const menu = await openMenu('Account: Mira Member (Member)');
    get.mockImplementation(async (url: string) => {
      if (url === '/v1/session') return GUEST_SESSION as never;
      if (url === '/v1/health') return HEALTH as never;
      throw new Error(`unexpected GET ${url}`);
    });

    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Sign out' }));
    await waitFor(() => expect(location.current).toBe('/'));
    expect(post).toHaveBeenCalledWith('/v1/auth/logout');
    const keys = invalidate.mock.calls.map(([filters]) => JSON.stringify(filters?.queryKey));
    for (const key of [['session'], ['capabilities'], ['jobs'], ['files']]) expect(keys).toContain(JSON.stringify(key));
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('a failed sign-out says so and keeps the user', async () => {
    const { post } = mockApi(MEMBER_SESSION);
    post.mockRejectedValue(new Error('offline'));
    renderApp('/app');
    const menu = await openMenu('Account: Mira Member (Member)');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Sign out' }));
    expect((await within(menu).findByRole('alert')).textContent).toBe('Could not sign out — try again.');
    expect(location.current).toBe('/app');
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

describe('sign-in notices (?signin= from the callback)', () => {
  it('not on the allowlist: a dismissible notice, and the params leave the URL', async () => {
    mockApi();
    renderApp('/app?signin=denied&reason=not_allowlisted&email=olga%40outside.e2e');
    const notice = await screen.findByTestId('signin-notice');
    expect(notice.textContent).toContain('olga@outside.e2e is not on the allowlist for this server');
    expect(notice.getAttribute('data-tone')).toBe('denied');
    await waitFor(() => expect(location.current).toBe('/app'));
    expect(screen.getByTestId('signin-notice')).toBeTruthy();

    fireEvent.click(within(notice).getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByTestId('signin-notice')).toBeNull();
  });

  it('unverified email and failure; other params stay', async () => {
    mockApi();
    renderApp('/app?signin=denied&reason=email_unverified');
    expect((await screen.findByTestId('signin-notice')).textContent).toContain('Your primary GitHub email is not verified');
    await waitFor(() => expect(location.current).toBe('/app'));
    cleanup();

    renderApp('/app/capabilities?signin=error&keep=1');
    const notice = await screen.findByTestId('signin-notice');
    expect(notice.textContent).toContain('Sign-in failed — try again');
    expect(notice.getAttribute('data-tone')).toBe('error');
    await waitFor(() => expect(location.current).toBe('/app/capabilities?keep=1'));
  });

  it('no notice without signin', () => {
    mockApi();
    renderApp('/app');
    expect(screen.queryByTestId('signin-notice')).toBeNull();
  });
});
