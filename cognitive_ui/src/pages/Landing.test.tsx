import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../api/client';
import { REDUCED_MOTION_QUERY } from '../hooks/useReducedMotion';
import { AppRoutes } from '../routes';
import { useAppStore } from '../store/useAppStore';
import { useChatStore } from '../store/useChatStore';
import { GUEST_SESSION, MEMBER_SESSION, OPERATOR_SESSION, SAMPLE } from '../test/fixtures';
import type { SessionInfo } from '../types';

// jsdom has no WebGL: the map panes render a stand-in.
vi.mock('../components/map/MapContainer', () => ({ MapContainer: () => <div data-testid="maplibre-map" /> }));

const HEALTH = { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: false, offline: true, provider: null, model: null } };

function mockSession(session: SessionInfo | Error) {
  return vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/session') {
      if (session instanceof Error) throw session;
      return session as never;
    }
    if (url === '/v1/health') return HEALTH as never;
    throw new Error(`unexpected GET ${url}`);
  });
}

function stubMotion(reduced: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: query === REDUCED_MOTION_QUERY ? reduced : false,
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
  delete document.documentElement.dataset.theme;
  stubMotion(false);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('/ routing', () => {
  it('shows the landing at once, before the session answers', () => {
    vi.spyOn(apiClient, 'get').mockImplementation(() => new Promise(() => {}));
    renderApp('/');
    expect(screen.getByTestId('landing')).toBeTruthy();
    expect(screen.queryByRole('navigation', { name: 'Main' })).toBeNull();
    expect(location.current).toBe('/');
  });

  it.each([
    ['guest', GUEST_SESSION],
    ['member', MEMBER_SESSION],
    ['operator', OPERATOR_SESSION],
  ])('%s: the landing, keeping the URL', async (_, session) => {
    const get = mockSession(session);
    renderApp('/?utm=x');
    await waitFor(() => expect(get).toHaveBeenCalledWith('/v1/session'));
    await act(async () => {});
    expect(screen.getByTestId('landing')).toBeTruthy();
    expect(location.current).toBe('/?utm=x');
  });

  it('shows the landing to a returning guest (stored conversations), leaving the open one as it is', async () => {
    mockSession(GUEST_SESSION);
    useChatStore.getState().addMessage({ kind: 'user', text: 'Earlier chat' });
    const active = useChatStore.getState().activeConversationId;
    renderApp('/');
    expect(screen.getByTestId('landing')).toBeTruthy();
    await act(async () => {});
    expect(location.current).toBe('/');
    expect(useChatStore.getState().activeConversationId).toBe(active);
  });

  it('shows the landing when the session request fails', async () => {
    mockSession(new Error('offline'));
    renderApp('/');
    await screen.findByTestId('landing');
  });

  it('"Open workspace" opens /app on the conversation that was open', async () => {
    mockSession(GUEST_SESSION);
    useChatStore.getState().addMessage({ kind: 'user', text: 'Earlier chat' });
    const active = useChatStore.getState().activeConversationId;
    renderApp('/');
    const link = screen.getByRole('link', { name: 'Open workspace' });
    expect(link.getAttribute('href')).toBe('/app');
    fireEvent.click(link);
    expect(location.current).toBe('/app');
    await act(async () => {});
    expect(useChatStore.getState().activeConversationId).toBe(active);
    expect(screen.getByRole('heading', { level: 1, name: 'Earlier chat' })).toBeTruthy();
  });

  describe.each([
    ['guest', GUEST_SESSION],
    ['member', MEMBER_SESSION],
  ])('a Workspace deep link, as a %s', (_, session) => {
    it('/?sample=1 opens /app and asks for the sample', async () => {
      mockSession(session);
      const post = vi.spyOn(apiClient, 'post').mockImplementation(() => new Promise(() => {}));
      renderApp('/?sample=1');
      await waitFor(() => expect(post).toHaveBeenCalledWith('/v1/sample'));
      expect(location.current).toBe('/app?sample=1');
      expect(screen.queryByTestId('landing')).toBeNull();
    });

    it('/?sample=0 opens /app?sample=0', async () => {
      mockSession(session);
      renderApp('/?sample=0');
      await waitFor(() => expect(location.current).toBe('/app?sample=0'));
      expect(screen.queryByTestId('landing')).toBeNull();
    });

    it('/?prompt= opens /app with the prompt in the composer', async () => {
      mockSession(session);
      renderApp(`/?${new URLSearchParams({ prompt: 'Map water change' })}`);
      await waitFor(() => expect(location.current).toBe('/app'));
      expect(useChatStore.getState().draft).toBe('Map water change');
      expect(screen.queryByTestId('landing')).toBeNull();
    });
  });
});

describe('the role line under "How it works"', () => {
  const GUEST_LINE = 'No account. Conversations stay in this browser for 24 h. Up to 5 analyses per hour.';
  const KEPT = 'Conversations stay in this browser and results on this server until you delete them. No analysis limit.';

  it("shows the guest's retention and quota once the session answers", async () => {
    mockSession({ ...GUEST_SESSION, kept_hours: 12, analyses: { ...GUEST_SESSION.analyses, limit: 3 } });
    renderApp('/');
    await screen.findByText('No account. Conversations stay in this browser for 12 h. Up to 3 analyses per hour.');
  });

  it('shows the guest line while the session loads', () => {
    vi.spyOn(apiClient, 'get').mockImplementation(() => new Promise(() => {}));
    renderApp('/');
    expect(screen.getByText(GUEST_LINE)).toBeTruthy();
  });

  it('shows the guest line when the session request fails', async () => {
    mockSession(new Error('offline'));
    renderApp('/');
    await screen.findByTestId('landing');
    await act(async () => {});
    expect(screen.getByText(GUEST_LINE)).toBeTruthy();
  });

  it.each([
    ['member', MEMBER_SESSION, 'Member'],
    ['operator', OPERATOR_SESSION, 'Operator'],
  ])('%s: their own line, with no guest quota', async (_, session, label) => {
    mockSession(session);
    renderApp('/');
    expect(screen.getByText(GUEST_LINE)).toBeTruthy();
    await screen.findByText(`${label}. ${KEPT}`);
    const landing = screen.getByTestId('landing');
    expect(landing.textContent).not.toMatch(/No account|analyses per hour/);
  });
});

describe('Landing', () => {
  async function openLanding() {
    mockSession(GUEST_SESSION);
    renderApp('/');
    return screen.findByTestId('landing');
  }

  it('renders in the stored theme, outside the app shell too', async () => {
    useAppStore.setState({ theme: 'dark' });
    await openLanding();
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(screen.queryByRole('navigation', { name: 'Main' })).toBeNull();
  });

  it("the full scene note on \"(bundled crop)\" is reachable by keyboard and shown on focus", async () => {
    await openLanding();
    const crop = screen.getByTestId('landing-caption-crop');
    expect(crop.tabIndex).toBe(0);
    const note = screen.getByTestId('landing-caption-note');
    expect(note.textContent).toBe(crop.getAttribute('title'));
    // Shown while focus is in the caption's sentence (CSS group-focus-within); screen readers get the sr-only copy.
    expect(note.className).toContain('hidden');
    expect(note.className).toContain('group-focus-within:block');
    expect(note.parentElement!.className).toContain('group');
    expect(note.getAttribute('aria-hidden')).toBe('true');
  });

  it("hero B: headline, both buttons as real links, the caption with the bundled crop's figure", async () => {
    await openLanding();
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('What changed on the ground?Ask in plain language.');
    expect(screen.getByText('Digital Twin for Land Cover · open-source research software')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Try with the Kakhovka sample' }).getAttribute('href')).toBe('/app?sample=1');
    expect(screen.getByRole('link', { name: 'Describe your own analysis' }).getAttribute('href')).toBe('/app?sample=0');
    expect(screen.getByTestId('landing-caption').textContent).toContain(
      'Example · Lower Dnipro, 1 → 9 June 2023 · +157 km² of water after the Kakhovka dam breach (bundled crop) · NASA HLS L30',
    );
    // The full scene in the hero differs: a tooltip on "(bundled crop)", read out to screen readers.
    const crop = screen.getByTestId('landing-caption-crop');
    expect(crop.getAttribute('title')).toMatch(/^Computed by the app on the bundled 768 × 768 px crop\. The full scene in this image shows more: \+428 km²/);
    expect(screen.getByTestId('landing-caption').textContent).toContain(`NASA HLS L30. ${crop.getAttribute('title')}`);
    const hero = screen.getByRole('img', { name: 'Water change layer over the lower Dnipro' });
    expect(hero.getAttribute('srcset')).toContain('_960.jpg 960w');
  });

  it('the Try button opens the Workspace with the sample', async () => {
    await openLanding();
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue(SAMPLE as never);
    fireEvent.click(screen.getByRole('link', { name: 'Try with the Kakhovka sample' }));
    expect(await screen.findByTestId('attachment-chip')).toBeTruthy();
    expect(post).toHaveBeenCalledWith('/v1/sample');
    await waitFor(() => expect(location.current).toBe('/app'));
  });

  it('three tier cards open Capabilities; the guest note follows the session', async () => {
    await openLanding();
    const tiers = screen.getAllByTestId('landing-tier');
    expect(tiers.map((t) => t.getAttribute('href'))).toEqual(['/app/capabilities', '/app/capabilities', '/app/capabilities']);
    expect(screen.getByText('No account. Conversations stay in this browser for 24 h. Up to 5 analyses per hour.')).toBeTruthy();
  });

  it('the footer: four logo slots (UMD and SNSF as text marks), funding, licence, GitHub, paper, imagery', async () => {
    await openLanding();
    const footer = within(screen.getByRole('contentinfo'));
    expect(footer.getByRole('img', { name: 'MMDA, Igor Sikorsky Kyiv Polytechnic Institute' }).getAttribute('src')).toBe('/brand/logo_mmda.png');
    expect(footer.getByRole('img', { name: 'University of Geneva' }).getAttribute('src')).toBe('/brand/logo_unige.png');
    expect(footer.getAllByTestId('logo-text-mark').map((m) => m.textContent)).toEqual(['UMD', 'SNSF']);
    expect(footer.getByText(/Grant 2023\.01\/0040/).textContent).toContain('Ukrainian-Swiss Joint Research Programme');
    expect(footer.getByText(/AGPL-3\.0/)).toBeTruthy();
    expect(footer.getByRole('link', { name: 'GitHub ↗' }).getAttribute('href')).toBe('https://github.com/IPT-MMDA/DT4LC-project');
    expect(footer.getByRole('link', { name: /Journal of Open Source Software/ }).getAttribute('href')).toBe(
      'https://github.com/IPT-MMDA/DT4LC-project/blob/master/paper.md',
    );
    expect(footer.getByText('Imagery: NASA HLS, Copernicus Sentinel-2')).toBeTruthy();
  });

  it("the demo quotes the bundled crop's figures, as the app computes them", async () => {
    await openLanding();
    const demo = within(screen.getByTestId('landing-demo'));
    expect(demo.getByText('13.9 km²')).toBeTruthy();
    expect(demo.getByText('170 km²')).toBeTruthy();
    expect(demo.getByText('+157 km²')).toBeTruthy();
    expect(demo.getByText('On the bundled crop, open water grew from 13.9 to 170 km² in eight days: 157 km² gained, 0.6 km² lost.')).toBeTruthy();
    expect(screen.getByTestId('landing-demo').textContent).not.toMatch(/428|256 km²/);
  });

  it('a step button scrubs the demo to that step', async () => {
    await openLanding();
    const demo = screen.getByTestId('landing-demo');
    expect(demo.dataset.mode).toBe('idle');
    fireEvent.click(screen.getByTestId('landing-step-2'));
    await waitFor(() => expect(demo.dataset.mode).toBe('scrub'));
    expect(demo.dataset.frame).toBe('2');
    expect(demo.style.getPropertyValue('--scrub')).toBe('-4.3s');
    expect(screen.getByTestId('landing-step-2').getAttribute('aria-current')).toBe('step');
    expect(screen.getByTestId('landing-step-1').getAttribute('aria-current')).toBeNull();
  });

  it('with reduced motion: the poster frame, no Play, and the steps still scrub', async () => {
    stubMotion(true);
    await openLanding();
    const demo = screen.getByTestId('landing-demo');
    expect(demo.dataset.mode).toBe('poster');
    expect(screen.getByTestId('landing-step-3').getAttribute('aria-current')).toBe('step');
    expect(screen.queryByRole('button', { name: 'Play the demo from the start' })).toBeNull();
    fireEvent.click(screen.getByTestId('landing-step-1'));
    await waitFor(() => expect(demo.dataset.mode).toBe('scrub'));
    expect(demo.dataset.frame).toBe('1');
  });

  it('Play restarts the demo from the start', async () => {
    await openLanding();
    const demo = screen.getByTestId('landing-demo');
    fireEvent.click(screen.getByRole('button', { name: 'Play the demo from the start' }));
    await waitFor(() => expect(demo.dataset.mode).toBe('play'));
    expect(demo.dataset.frame).toBe('0');
  });
});
