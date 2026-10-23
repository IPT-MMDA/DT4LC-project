import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from './api/client';
import type { JobSummary } from './api/library';
import { screenTitle } from './components/shell/navigation';
import { useLayout } from './hooks/useMediaQuery';
import { Library } from './pages/Library';
import { AppRoutes } from './routes';
import { useAppStore } from './store/useAppStore';
import { useChatStore } from './store/useChatStore';
import DEMO_JSON from './test/fixtures/capabilities/demo-guest.json';
import { GUEST_SESSION } from './test/fixtures';
import type { Conversation } from './types';

// Phone (< 768 px), tablet (768–1023 px) and desktop layouts. matchMedia answers width
// queries for a simulated window width, and resize() fires their change events.

// jsdom has no WebGL: the map is a stand-in.
vi.mock('./components/map/MapContainer', () => ({ MapContainer: () => <div data-testid="maplibre-map" /> }));

const HEALTH = { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: false, offline: true, provider: null, model: null } };

let width = 1440;
const listeners = new Set<() => void>();
function widthMatches(query: string) {
  const min = /min-width:\s*(\d+)px/.exec(query);
  const max = /max-width:\s*(\d+)px/.exec(query);
  if (!min && !max) return false;
  return (!min || width >= Number(min[1])) && (!max || width <= Number(max[1]));
}
function stubWidth(initial: number) {
  width = initial;
  listeners.clear();
  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return widthMatches(query);
    },
    media: query,
    addEventListener: (_: string, fn: () => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
  }));
}
function resize(next: number) {
  width = next;
  act(() => listeners.forEach((fn) => fn()));
}

function mockApi() {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/session') return GUEST_SESSION as never;
    if (url === '/v1/health') return HEALTH as never;
    if (url === '/v1/capabilities') return DEMO_JSON as never;
    throw new Error(`unexpected GET ${url}`);
  });
}

const location = { current: '' };
function LocationProbe() {
  const loc = useLocation();
  location.current = loc.pathname;
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

const conversation = (id: string, title: string): Conversation => ({
  id,
  title,
  createdAt: '2026-10-23T10:00:00.000Z',
  updatedAt: '2026-10-23T10:00:00.000Z',
  messages: [],
});

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({
    conversations: [conversation('c1', 'Kakhovka reservoir — water loss'), conversation('c2', 'Snow in Brienz')],
    activeConversationId: 'c1',
    pendingAttachments: [],
    draft: '',
  });
  useAppStore.setState({ theme: 'light', drawnRegions: [], selectedRegion: null, mapLayers: [], imagery: null });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('useLayout', () => {
  it('switches at 768 and 1024 px, as Tailwind md and lg, and follows a resize', () => {
    stubWidth(390);
    const { result } = renderHook(() => useLayout());
    expect(result.current).toBe('phone');
    for (const [w, layout] of [[767, 'phone'], [768, 'tablet'], [834, 'tablet'], [1023, 'tablet'], [1024, 'desktop'], [1440, 'desktop'], [390, 'phone']] as const) {
      resize(w);
      expect(result.current, `${w} px`).toBe(layout);
    }
  });

  it('is desktop without matchMedia', () => {
    vi.stubGlobal('matchMedia', undefined);
    expect(renderHook(() => useLayout()).result.current).toBe('desktop');
  });
});

describe('screenTitle', () => {
  it("is the open conversation's name in the Workspace, else the screen's", () => {
    expect(screenTitle('/app', 'Kakhovka reservoir — water loss')).toBe('Kakhovka reservoir — water loss');
    expect(screenTitle('/app')).toBe('Chat');
    expect(screenTitle('/app/map')).toBe('Map');
    expect(screenTitle('/app/library')).toBe('Library');
    expect(screenTitle('/app/results/a1')).toBe('Result');
    expect(screenTitle('/app/capabilities')).toBe('Can do');
    expect(screenTitle('/app/settings')).toBe('Settings');
  });
});

describe('the shell on a phone', () => {
  it('has a top bar with the conversation as its heading and Map ↗, no rail and no list beside the chat', () => {
    stubWidth(390);
    mockApi();
    renderApp('/app');
    const bar = screen.getByTestId('phone-bar');
    expect(within(bar).getByRole('heading', { level: 1 }).textContent).toBe('Kakhovka reservoir — water loss');
    expect(within(bar).getByRole('link', { name: 'Map ↗' }).getAttribute('href')).toBe('/app/map');
    expect(screen.queryByRole('navigation', { name: 'Main' })).toBeNull();
    expect(screen.queryByRole('complementary', { name: 'Conversations' })).toBeNull();
    expect(screen.getByRole('main', { name: 'Chat' })).toBeTruthy();
  });

  it('the menu opens a drawer with the destinations and the conversations; Escape closes it and focus returns', () => {
    stubWidth(390);
    mockApi();
    renderApp('/app');
    const menu = screen.getByRole('button', { name: 'Open menu' });
    expect(menu.getAttribute('aria-expanded')).toBe('false');

    fireEvent.click(menu);
    expect(menu.getAttribute('aria-expanded')).toBe('true');
    const drawer = screen.getByRole('dialog', { name: 'Menu' });
    expect(menu.getAttribute('aria-controls')).toBe(drawer.id);
    expect(document.activeElement).toBe(within(drawer).getByRole('button', { name: 'Close menu' }));
    const nav = within(drawer).getByRole('navigation', { name: 'Main' });
    expect(within(nav).getAllByRole('link').map((a) => a.textContent)).toEqual(['Chat', 'Map', 'Library', 'Can do']);
    expect(within(nav).getByRole('link', { name: 'Chat' }).getAttribute('aria-current')).toBe('page');
    const list = within(drawer).getByRole('complementary', { name: 'Conversations' });
    expect(within(list).getAllByTestId('conversation')).toHaveLength(2);

    fireEvent.keyDown(drawer, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'Menu' })).toBeNull();
    expect(menu.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(menu);
  });

  it('keeps Tab inside the drawer, both ways, and gives focus back to the menu button when another conversation closes it', async () => {
    stubWidth(390);
    mockApi();
    renderApp('/app');
    const menu = screen.getByRole('button', { name: 'Open menu' });
    menu.focus();
    fireEvent.click(menu);
    const drawer = screen.getByRole('dialog', { name: 'Menu' });
    await within(drawer).findByRole('button', { name: 'Account: Guest' });
    const close = within(drawer).getByRole('button', { name: 'Close menu' });
    const theme = within(drawer).getByRole('button', { name: 'Dark theme' });
    expect(document.activeElement).toBe(close);

    // The theme button is the drawer's last control: Tab wraps to the first, Shift+Tab back.
    const home = within(drawer).getByRole('link', { name: 'DT, DT4LC home' });
    theme.focus();
    fireEvent.keyDown(theme, { key: 'Tab' });
    expect(document.activeElement).toBe(home);
    fireEvent.keyDown(home, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(theme);
    // Between the ends Tab is the browser's.
    close.focus();
    expect(fireEvent.keyDown(close, { key: 'Tab' })).toBe(true);
    // Focus that left the drawer is pulled back in.
    menu.focus();
    fireEvent.keyDown(menu, { key: 'Tab' });
    expect(document.activeElement).toBe(home);

    fireEvent.click(within(drawer).getByRole('button', { name: 'Snow in Brienz' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Open menu' })));
  });

  it('a destination, the current one too, another conversation and Close each close the drawer', () => {
    stubWidth(390);
    mockApi();
    renderApp('/app');
    const open = () => {
      fireEvent.click(screen.getByRole('button', { name: 'Open menu' }));
      return screen.getByRole('dialog', { name: 'Menu' });
    };

    fireEvent.click(within(open()).getByRole('button', { name: 'Snow in Brienz' }));
    expect(useChatStore.getState().activeConversationId).toBe('c2');
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(within(screen.getByTestId('phone-bar')).getByRole('heading').textContent).toBe('Snow in Brienz');

    fireEvent.click(within(open()).getByRole('link', { name: 'Library' }));
    expect(location.current).toBe('/app/library');
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.getByTestId('phone-bar').textContent).toContain('Library');

    fireEvent.click(within(open()).getByRole('link', { name: 'Library' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    // Not in the Workspace: no conversations in the menu.
    expect(within(open()).queryByRole('complementary', { name: 'Conversations' })).toBeNull();
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Close menu' }));
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it("Escape in the drawer's account menu closes only the account menu", async () => {
    stubWidth(390);
    mockApi();
    renderApp('/app');
    fireEvent.click(screen.getByRole('button', { name: 'Open menu' }));
    const drawer = screen.getByRole('dialog', { name: 'Menu' });
    const account = await within(drawer).findByRole('button', { name: 'Account: Guest' });
    fireEvent.click(account);
    expect(screen.getByRole('menu', { name: 'Account' }).className).toContain('bottom-full');

    fireEvent.keyDown(account, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('menu', { name: 'Account' })).toBeNull());
    expect(screen.getByRole('dialog', { name: 'Menu' })).toBeTruthy();
  });

  it('turns back into the rail when the window widens', () => {
    stubWidth(390);
    mockApi();
    renderApp('/app');
    fireEvent.click(screen.getByRole('button', { name: 'Open menu' }));
    resize(1280);
    expect(screen.queryByTestId('phone-bar')).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
    expect(screen.getByRole('complementary', { name: 'Conversations' })).toBeTruthy();
  });
});

describe('the Workspace on a tablet', () => {
  it('keeps the rail and hides and shows the conversation list', () => {
    stubWidth(834);
    mockApi();
    renderApp('/app');
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeTruthy();
    const list = screen.getByRole('complementary', { name: 'Conversations' });

    fireEvent.click(within(list).getByRole('button', { name: 'Hide conversations' }));
    expect(screen.queryByRole('complementary', { name: 'Conversations' })).toBeNull();
    const show = screen.getByRole('button', { name: 'Show conversations' });
    expect(show.getAttribute('aria-expanded')).toBe('false');

    fireEvent.click(show);
    expect(screen.getByRole('complementary', { name: 'Conversations' })).toBeTruthy();
  });

  it('has no collapse button on a desktop', () => {
    stubWidth(1280);
    mockApi();
    renderApp('/app');
    expect(screen.queryByRole('button', { name: 'Hide conversations' })).toBeNull();
  });
});

describe('the landing nav', () => {
  beforeEach(() => {
    useChatStore.setState({ conversations: [], activeConversationId: null });
  });

  it('collapses behind a Menu button on a phone, which opens and closes it', async () => {
    stubWidth(390);
    mockApi();
    renderApp('/');
    await screen.findByTestId('landing');
    expect(screen.queryByRole('navigation', { name: 'Landing' })).toBeNull();
    const menu = screen.getByRole('button', { name: 'Menu' });
    expect(menu.getAttribute('aria-expanded')).toBe('false');

    fireEvent.click(menu);
    expect(menu.getAttribute('aria-expanded')).toBe('true');
    const nav = screen.getByRole('navigation', { name: 'Landing' });
    expect(menu.getAttribute('aria-controls')).toBe(nav.id);
    expect(within(nav).getAllByRole('link').map((a) => a.textContent)).toEqual(['Capabilities', 'How it works', 'About', 'GitHub ↗']);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('navigation', { name: 'Landing' })).toBeNull();

    fireEvent.click(menu);
    fireEvent.click(within(screen.getByRole('navigation', { name: 'Landing' })).getByRole('link', { name: 'How it works' }));
    expect(screen.queryByRole('navigation', { name: 'Landing' })).toBeNull();
    expect(menu.getAttribute('aria-expanded')).toBe('false');
  });

  it('puts the caption under the buttons on a phone, and shows the nav without a menu on a tablet', async () => {
    stubWidth(390);
    mockApi();
    const { unmount } = renderApp('/');
    const caption = await screen.findByTestId('landing-caption');
    const buttons = screen.getByRole('link', { name: 'Describe your own analysis' }).parentElement!;
    // In the hero's column, after the buttons.
    expect(caption.parentElement).toBe(buttons.parentElement);
    expect(buttons.compareDocumentPosition(caption) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByRole('heading', { level: 1 }).className).toContain('max-md:text-hero-phone');
    unmount();

    stubWidth(834);
    renderApp('/');
    await screen.findByTestId('landing');
    expect(screen.queryByRole('button', { name: 'Menu' })).toBeNull();
    expect(within(screen.getByRole('navigation', { name: 'Landing' })).getAllByRole('link')).toHaveLength(4);
    expect(screen.getByTestId('landing-caption').parentElement).toBe(screen.getByTestId('landing-hero'));
  });
});

describe('map mode on a phone', () => {
  it('puts the imagery panel in a sheet under the map that opens and closes', async () => {
    stubWidth(390);
    mockApi();
    renderApp('/app/map');
    const panel = screen.getByRole('complementary', { name: 'Imagery' });
    const map = screen.getByRole('main', { name: 'Map' });
    expect(map.compareDocumentPosition(panel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const toggle = within(panel).getByRole('button', { name: 'Imagery panel' });
    const body = document.getElementById(toggle.getAttribute('aria-controls')!)!;
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(body.classList.contains('hidden')).toBe(true);
    expect(within(panel).getByRole('link', { name: 'Back to chat' })).toBeTruthy();

    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(body.classList.contains('hidden')).toBe(false);
    expect(within(body).getByRole('group', { name: 'Region' })).toBeTruthy();

    fireEvent.click(toggle);
    expect(body.classList.contains('hidden')).toBe(true);
  });

  it('keeps the side panel without a sheet toggle on a tablet', () => {
    stubWidth(834);
    mockApi();
    renderApp('/app/map');
    const panel = screen.getByRole('complementary', { name: 'Imagery' });
    expect(within(panel).queryByRole('button', { name: 'Imagery panel' })).toBeNull();
    expect(panel.compareDocumentPosition(screen.getByRole('main', { name: 'Map' })) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});

describe('the Library on a phone', () => {
  const card = { title: 'Water change · NDWI difference', analysis: { id: 'algorithms/change-detection', label: 'Change detection' }, kind: 'change' as const, thumbnail_url: null, duration_s: 48, error: null };
  const JOBS: JobSummary[] = [{ id: 'a1', status: 'completed', prompt: 'Compare water', created_at: new Date().toISOString(), card }];

  function renderLibrary() {
    vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
      if (url === '/v1/session') return GUEST_SESSION as never;
      throw new Error(`unexpected GET ${url}`);
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (url.startsWith('/v1/jobs')) return new Response(JSON.stringify({ jobs: JOBS, total: 1, limit: 100, offset: 0 }));
        if (url === '/v1/files') return new Response(JSON.stringify({ ok: true, files: [], count: 0 }));
        throw new Error(`unexpected fetch ${url}`);
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <Library />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  }

  it('lists cards with the thumbnail, name, type, status, when and the row action; no table', async () => {
    stubWidth(390);
    renderLibrary();
    const row = await screen.findByTestId('library-row');
    expect(row.tagName).toBe('LI');
    expect(screen.queryByRole('table')).toBeNull();
    const r = within(row);
    expect(r.getByTestId('library-thumbnail')).toBeTruthy();
    expect(r.getByTestId('library-name').textContent).toBe('Water change · NDWI difference');
    expect(r.getByTestId('library-type').textContent).toBe('Change detection');
    expect(r.getByTestId('library-status').textContent).toContain('Done');
    expect(r.getByTestId('library-when').textContent).toMatch(/^today \d\d:\d\d$/);
    expect(r.getByRole('button', { name: 'Map' })).toBeTruthy();
    expect(r.getByRole('button', { name: `More for ${card.title}` })).toBeTruthy();
    expect(r.queryByTestId('library-conversation')).toBeNull();
  });

  it('keeps the table on a tablet', async () => {
    stubWidth(834);
    renderLibrary();
    const row = await screen.findByTestId('library-row');
    expect(row.tagName).toBe('TR');
    expect(screen.getByRole('table')).toBeTruthy();
  });
});
