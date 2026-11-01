import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../api/client';
import type { Capabilities } from '../api/hooks/useCapabilities';
import { AppRoutes } from '../routes';
import { useAppStore } from '../store/useAppStore';
import { useChatStore } from '../store/useChatStore';
import DEMO_JSON from '../test/fixtures/capabilities/demo-guest.json';
import OFFLINE_JSON from '../test/fixtures/capabilities/offline-member.json';
import { GUEST_SESSION, MEMBER_SESSION } from '../test/fixtures';
import type { Attachment, SessionInfo } from '../types';

// Real GET /v1/capabilities from the E2E API (example_prompts trimmed):
// demo-guest: E2E_SERVER=demo E2E_LLM=fake E2E_GUEST_LIMIT=5 (guest, Delineate staged as installed, EE configured);
// offline-member: E2E_DEV_ROLE=member E2E_LLM=none.
const DEMO = DEMO_JSON as unknown as Capabilities;
const OFFLINE = OFFLINE_JSON as unknown as Capabilities;

// jsdom has no WebGL: the docked map renders a stand-in.
vi.mock('../components/map/MapContainer', () => ({ MapContainer: () => <div data-testid="maplibre-map" /> }));

const HEALTH = { ok: true, service: 'DT4LC', version: '0.6.0', assistant: { connected: true, offline: false, provider: 'ollama', model: 'fake' } };

function mockApi(doc: Capabilities, session: SessionInfo = GUEST_SESSION) {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/capabilities') return doc as never;
    if (url === '/v1/session') return session as never;
    if (url === '/v1/health') return HEALTH as never;
    throw new Error(`unexpected GET ${url}`);
  });
  return vi.spyOn(apiClient, 'post').mockImplementation(async (url: string) => {
    throw new Error(`unexpected POST ${url}`);
  });
}

const location = { current: '' };
function LocationProbe() {
  const loc = useLocation();
  location.current = loc.pathname + loc.search;
  return null;
}

function renderAt(doc: Capabilities, session?: SessionInfo, url = '/app/capabilities') {
  const post = mockApi(doc, session);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[url]}>
        <AppRoutes />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return post;
}

const tier = (id: string) => screen.getByTestId(`tier-${id}`);
const statuses = (el: HTMLElement) =>
  within(el)
    .queryAllByTestId('capability-status')
    .map((s) => [s.textContent, s.dataset.hue]);
const builtin = (id: string) => screen.getAllByTestId('builtin-item').find((el) => el.dataset.id === id)!;
const promptIn = (el: HTMLElement) => within(el).getByTestId('example-prompt') as HTMLButtonElement;

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], draft: '' });
  useAppStore.setState({ theme: 'light' });
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: query.includes('min-width: 1440px'),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('Capabilities page, public demo', () => {
  it('renders the title, the live summary line and the six tiers in order', async () => {
    renderAt(DEMO);
    await screen.findByTestId('tier-ask');

    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('What DT4LC can do · on this server');
    expect(screen.getByTestId('capabilities-summary').textContent).toBe(
      'Live for this installation: public demo · guest access · fake connected · Delineate-Anything installed · Earth Engine configured.',
    );
    const main = screen.getByTestId('capabilities-page');
    const order = within(main)
      .getAllByTestId(/^tier-/)
      .map((el) => el.dataset.testid);
    expect(order).toEqual(['tier-ask', 'tier-composed', 'tier-builtin', 'tier-models', 'tier-data', 'tier-coming']);
    expect(screen.queryByTestId('capabilities-banner')).toBeNull();
    expect(order.map((id) => screen.getByTestId(id!).dataset.dimmed ?? 'false')).toEqual(Array(6).fill('false'));
  });

  it('shows the assistant connected, Composed badges, recipes and "Needs the assistant"', async () => {
    renderAt(DEMO);
    expect(statuses(await screen.findByTestId('tier-ask'))).toEqual([['Assistant · fake · connected', 'done']]);
    expect(within(tier('ask')).getAllByTestId('example-prompt').map((b) => b.textContent)).toEqual([
      '"Why is NDWI better than NDVI for water?"',
      '"Explain the last result in two sentences"',
      '"What would you do differently with Landsat?"',
    ]);
    const composed = tier('composed');
    expect(within(composed).getAllByText('Composed')).toHaveLength(4);
    expect(within(composed).getAllByText('Needs the assistant')).toHaveLength(4);
    expect(within(composed).getByText('NDMI = (B5 − B6)/(B5 + B6) · NBR · SAVI · NDBI · custom')).toBeTruthy();
  });

  it('groups the built-in analyses, with inputs, outputs, time and a status line each', async () => {
    renderAt(DEMO);
    await screen.findByTestId('tier-builtin');
    expect(within(tier('builtin')).getAllByRole('heading', { level: 3 }).filter((h) => h.className.includes('uppercase')).map((h) => h.textContent)).toEqual([
      'Spectral indices',
      'Change, classification, statistics',
      'Model-based',
    ]);
    expect(within(screen.getByTestId('builtin-group-spectral')).getAllByTestId('builtin-item').map((el) => el.dataset.id)).toEqual([
      'algorithms/ndvi',
      'algorithms/evi',
      'algorithms/ndwi',
      'algorithms/ndsi',
    ]);
    const ndvi = builtin('algorithms/ndvi');
    expect(within(ndvi).getByText('(NIR−Red)/(NIR+Red)')).toBeTruthy();
    expect(within(ndvi).getByText('Red, NIR')).toBeTruthy();
    expect(within(ndvi).getByText('Index map, legend, stats')).toBeTruthy();
    expect(within(ndvi).getByText('~10 s')).toBeTruthy();
    expect(statuses(ndvi)).toEqual([['Runs without the assistant', 'done']]);
  });

  it('dependent analyses inherit their model status: installed runs, gated for a guest is disabled', async () => {
    renderAt(DEMO);
    await screen.findByTestId('tier-builtin');

    const fields = builtin('models/delineate-anything');
    expect(within(fields).getByText('Delineate-Anything')).toBeTruthy();
    expect(statuses(fields)).toEqual([['Runs without the assistant', 'done']]);
    expect(promptIn(fields).disabled).toBe(false);

    const prithvi = builtin('models/prithvi-reconstruction');
    expect(within(prithvi).getByText('Prithvi EO v1')).toBeTruthy();
    expect(statuses(prithvi)).toEqual([['Not available for guests — ask an operator', 'muted']]);
    expect(promptIn(prithvi).disabled).toBe(true);
  });

  it('shows the models and data sources with their statuses, licence links and the guest note', async () => {
    renderAt(DEMO);
    expect(statuses(await screen.findByTestId('tier-models'))).toEqual([
      ['Installed on this server', 'done'],
      ['Not available for guests — ask an operator', 'muted'],
    ]);
    const licence = within(tier('models')).getByRole('link', { name: 'AGPL-3.0' });
    expect(licence.getAttribute('href')).toBe('https://github.com/Lavreniuk/Delineate-Anything');
    expect(within(tier('models')).getByText(/85 MB ·/)).toBeTruthy();

    expect(statuses(tier('data'))).toEqual([
      ['Always', 'done'],
      ['Always', 'done'],
      ['Configured', 'done'],
    ]);
    expect(within(tier('data')).getByText('Guests: up to 5 analyses per hour, results kept 24 h in this browser, no account needed.')).toBeTruthy();
    expect(within(tier('data')).getByText('kakhovka_hls_l30 · HLS L30 · 2023-06-01 → 06-09')).toBeTruthy();
  });
});

describe('Capabilities page, offline laptop', () => {
  it('shows the banner and greys tiers 1 and 2 without hiding them', async () => {
    renderAt(OFFLINE, MEMBER_SESSION);
    const banner = await screen.findByTestId('capabilities-banner');
    expect(banner.textContent).toBe(OFFLINE.banner);

    expect(tier('ask').dataset.dimmed).toBe('true');
    expect(tier('composed').dataset.dimmed).toBe('true');
    // Dimmed by colour, not opacity, so the text keeps AA contrast.
    expect(tier('ask').className).not.toMatch(/opacity/);
    expect(tier('ask').className).toContain('!text-muted');
    for (const id of ['builtin', 'models', 'data']) expect(tier(id).dataset.dimmed).toBe('false');
    // Greyed, not hidden, and still usable.
    expect(within(tier('ask')).getAllByTestId('example-prompt')).toHaveLength(3);
    expect(within(tier('composed')).getAllByTestId('example-prompt').every((b) => !(b as HTMLButtonElement).disabled)).toBe(true);
    expect(statuses(tier('ask'))).toEqual([['Assistant offline — limited answers', 'canc']]);
  });

  it('a member sees models that download on first use, and their analyses inherit it', async () => {
    renderAt(OFFLINE, MEMBER_SESSION);
    expect(statuses(await screen.findByTestId('tier-models'))).toEqual([
      ['Downloads on first use · 85 MB · ~14 s · AGPL-3.0', 'run'],
      ['Downloads on first use · 450 MB · ~75 s · Apache-2.0', 'run'],
    ]);
    const prithvi = builtin('models/prithvi-reconstruction');
    expect(statuses(prithvi)).toEqual([['Downloads on first use · 450 MB · ~75 s · Apache-2.0', 'run']]);
    expect(promptIn(prithvi).disabled).toBe(false);
    expect(statuses(tier('data'))[2]).toEqual(['Not configured', 'muted']);
    // The Data note follows the role: a member has no guest limits.
    expect(within(tier('data')).getByText('Member on a local server: no analysis limit; results are kept until you delete them.')).toBeTruthy();
    expect(within(tier('data')).queryByText(/^Guests:/)).toBeNull();
    expect(screen.getByTestId('capabilities-summary').textContent).toBe(
      'Live for this installation: member account · assistant offline · models download on first use · no Earth Engine.',
    );
  });
});

describe('Capabilities page, controls', () => {
  it('each tier chip is a button that scrolls to its tier', async () => {
    const scrolled: (string | undefined)[] = [];
    Element.prototype.scrollIntoView = vi.fn(function (this: HTMLElement) {
      scrolled.push(this.dataset.testid);
    });
    renderAt(DEMO);
    const chips = within(await screen.findByRole('navigation', { name: 'Tiers' })).getAllByRole('button');
    expect(chips.map((c) => c.textContent)).toEqual(['Ask', 'Composed', 'Built-in', 'Models', 'Data']);
    chips.forEach((chip) => fireEvent.click(chip));
    expect(scrolled).toEqual(['tier-ask', 'tier-composed', 'tier-builtin', 'tier-models', 'tier-data']);
  });

  it('Coming is collapsed with a Show toggle', async () => {
    renderAt(DEMO);
    const coming = await screen.findByTestId('tier-coming');
    const toggle = within(coming).getByRole('button');
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(toggle.textContent).toContain('ConvLSTM NDVI forecast · PINN-LINDA nowcasting · VI forecasting DTI · Runoff ensemble');
    expect(toggle.textContent).toContain('Show');
    expect(within(coming).queryByText(/Predicts vegetation health/)).toBeNull();

    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(toggle.textContent).toContain('Hide');
    expect(within(coming).getByText(/Predicts vegetation health/)).toBeTruthy();
    // Four entries, the runoff ensemble last.
    const items = coming.querySelectorAll('#tier-coming-items > div');
    expect([...items].map((i) => i.firstElementChild?.textContent)).toEqual([
      'ConvLSTM NDVI forecast',
      'PINN-LINDA nowcasting',
      'VI forecasting DTI',
      'Runoff ensemble',
    ]);
  });

  it('an example prompt opens the Workspace with it in the composer, not sent, files kept', async () => {
    const mine: Attachment = { id: 'up1', filename: 'mine.tif', path: '/u/up1_mine.tif' };
    useChatStore.setState({ pendingAttachments: [mine] });
    const post = renderAt(DEMO);
    await screen.findByTestId('tier-builtin');

    fireEvent.click(promptIn(builtin('algorithms/change-detection')));

    const input = (await screen.findByLabelText('Message')) as HTMLTextAreaElement;
    await waitFor(() => expect(location.current).toBe('/app'));
    expect(input.value).toBe('Compare water before and after the dam breach');
    expect(useChatStore.getState().pendingAttachments).toEqual([mine]);
    expect(useChatStore.getState().conversations).toEqual([]);
    expect(post).not.toHaveBeenCalled();
  });

  it('a gated prompt does nothing', async () => {
    renderAt(DEMO);
    await screen.findByTestId('tier-builtin');
    fireEvent.click(promptIn(builtin('models/prithvi-reconstruction')));
    expect(location.current).toBe('/app/capabilities');
    expect(useChatStore.getState().draft).toBe('');
  });
});
