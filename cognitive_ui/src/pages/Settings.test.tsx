import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import apiClient, { ApiError } from '../api/client';
import type { SettingsDoc } from '../api/settings';
import DOC from '../test/fixtures/settings/operator.json';
import TEST_RESULT from '../test/fixtures/settings/assistant-test.json';
import { Settings } from './Settings';

const doc = () => structuredClone(DOC) as SettingsDoc;

function mockApi(settings: SettingsDoc = doc()) {
  const get = vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/settings') return settings as never;
    if (url.startsWith('/v1/ml-models/')) return { id: url.split('/').pop(), status: 'downloading', download_progress: { percent: 42 } } as never;
    throw new Error(`unexpected GET ${url}`);
  });
  const put = vi.spyOn(apiClient, 'put').mockImplementation(async () => settings as never);
  const post = vi.spyOn(apiClient, 'post').mockImplementation(async (url: string) => {
    if (url === '/v1/settings/assistant/test') return TEST_RESULT as never;
    if (url.endsWith('/download')) return { status: 'downloading' } as never;
    throw new Error(`unexpected POST ${url}`);
  });
  return { get, put, post };
}

function renderSettings() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const invalidate = vi.spyOn(client, 'invalidateQueries');
  const view = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/app/settings']}>
        <Settings />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { ...view, invalidate };
}

const tab = (name: string) => fireEvent.click(within(screen.getByRole('navigation', { name: 'Settings sections' })).getByRole('button', { name: new RegExp(`^${name}`) }));
const section = (id: string) => screen.getByTestId(`settings-section-${id}`);
const field = (path: string) => document.querySelector(`[data-testid="setting-field"][data-path="${path}"]`) as HTMLElement;
const save = () => fireEvent.click(screen.getByRole('button', { name: 'Save' }));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('Settings: the five sections from GET /v1/settings', () => {
  it('Assistant: the connected line, the order with real buttons, key status, models', async () => {
    mockApi();
    renderSettings();
    const s = await screen.findByTestId('settings-section-assistant');
    expect(screen.getByRole('heading', { name: 'Settings · this server' })).toBeTruthy();
    expect(within(s).getByRole('heading', { name: 'Assistant' })).toBeTruthy();
    expect(within(s).getByTestId('assistant-connected').textContent).toBe('Connected · Ollama · fake');

    const ordered = within(within(s).getByRole('list', { name: 'Providers in order' })).getAllByTestId('provider');
    expect(ordered.map((p) => p.dataset.name)).toEqual(['ollama', 'mistral']);
    const others = within(within(s).getByRole('list', { name: 'Providers not in the order' })).getAllByTestId('provider');
    expect(others.map((p) => p.dataset.name)).toEqual(['gemini', 'groq', 'apertus', 'anthropic', 'openai']);

    const gemini = others[0];
    expect(within(gemini).getByTestId('key-status').textContent).toBe('Key present');
    expect(within(gemini).getByRole('textbox', { name: 'Gemini model 1' })).toHaveProperty('value', 'gemini-3.5-flash-lite');
    expect(within(gemini).getByRole('textbox', { name: 'Gemini model 2' })).toHaveProperty('value', 'gemini-3.5-flash');
    expect(within(gemini).getByRole('button', { name: 'Add Gemini to the order' })).toBeTruthy();
    expect(within(ordered[1]).getByTestId('key-status').textContent).toBe('No key');
    // A single-model provider has one input.
    expect(within(ordered[0]).getByRole('textbox', { name: 'Ollama model' })).toHaveProperty('value', 'fake');
    expect(within(ordered[0]).getByText('No key needed')).toBeTruthy();
    expect(within(ordered[0]).getByRole('switch', { name: 'Ollama enabled' })).toHaveProperty('checked', true);
    expect(within(ordered[0]).getByRole('button', { name: 'Move Ollama up' })).toHaveProperty('disabled', true);
    expect(within(ordered[0]).getByRole('button', { name: 'Move Ollama down' })).toHaveProperty('disabled', false);
    expect(within(s).getAllByTestId('setting-source')[0].textContent).toBe('from .env');
  });

  it('Models, Access, Data sources and Server', async () => {
    mockApi();
    renderSettings();
    await screen.findByTestId('settings-section-assistant');

    tab('Models');
    const models = section('models');
    const prithvi = within(models).getAllByTestId('settings-model')[0];
    expect(prithvi.textContent).toContain('Prithvi EO v1 (100M)');
    expect(prithvi.textContent).toContain('Not installed');
    expect(prithvi.textContent).toContain('450.0 MB');
    expect(within(prithvi).getByRole('link', { name: 'Apache-2.0' }).getAttribute('href')).toContain('huggingface.co');
    expect(within(prithvi).getByRole('button', { name: 'Pre-download Prithvi EO v1 (100M)' })).toBeTruthy();
    expect(within(models).getByRole('combobox', { name: 'Downloads on first use' })).toHaveProperty('value', 'members');

    tab('Access');
    const access = section('access');
    expect(within(access).getByTestId('sign-in-status').textContent).toBe('Sign-in not configured');
    expect(within(access).getByRole('spinbutton', { name: 'Guest analyses per hour' })).toHaveProperty('value', '1000');
    expect(within(access).getByText('0–1000')).toBeTruthy();
    expect(within(access).getByText('1–720')).toBeTruthy();
    expect(within(access).getByRole('textbox', { name: 'Add to Operator emails' })).toBeTruthy();

    tab('Data sources');
    const data = section('data_sources');
    expect(within(data).getByTestId('ee-status').textContent).toBe('Not configured');
    expect(data.textContent).toContain('Project id missing · key file missing');
    expect(data.textContent).toContain('/tmp/dt4lc-e2e-state/uploads');
    expect(data.textContent).toContain('0 files · 0 B');
    expect(data.textContent).toContain('kakhovka_hls_l30');
    expect(data.textContent).toContain('Bundled');

    tab('Server');
    const server = section('server');
    expect(server.textContent).toContain('0.5.1');
    expect(server.textContent).toContain('Local server');
    expect(server.textContent).toContain('0 in total · 0 queued · 3 workers');
    expect(within(server).getAllByTestId('storage-row').map((r) => r.dataset.key)).toEqual(['cache', 'outputs', 'temp', 'uploads', 'models']);
  });

  it('read-only sections and fields render no inputs', async () => {
    const settings = doc();
    settings.access.editable.guest_analyses_per_hour = false;
    settings.assistant.providers.ollama.editable.models = false;
    mockApi(settings);
    renderSettings();
    const assistant = await screen.findByTestId('settings-section-assistant');
    expect(within(assistant).queryByRole('textbox', { name: 'Ollama model' })).toBeNull();
    const ollama = within(assistant).getAllByTestId('provider')[0];
    expect(within(ollama).getByText('fake')).toBeTruthy();

    tab('Access');
    expect(within(section('access')).queryByRole('spinbutton', { name: 'Guest analyses per hour' })).toBeNull();
    expect(section('access').textContent).toContain('Guest analyses per hour: 1000');
    for (const id of ['Data sources', 'Server']) {
      tab(id);
      const s = screen.getByRole('region', { name: id });
      expect(s.querySelectorAll('input, select, textarea').length).toBe(0);
    }
  });

  it('never shows a key value, and never asks for one', async () => {
    const settings = doc();
    // As if a key leaked into the document: the page renders only the fields it knows.
    Object.assign(settings.assistant.providers.gemini, { api_key: 'sk-e2e-secret-key-123' });
    mockApi(settings);
    const { container } = renderSettings();
    await screen.findByTestId('settings-section-assistant');
    for (const id of ['Assistant', 'Models', 'Access', 'Data sources', 'Server']) {
      tab(id);
      expect(container.innerHTML).not.toContain('sk-e2e-secret-key-123');
      expect(container.querySelectorAll('input[type="password"]').length).toBe(0);
      expect(screen.queryByLabelText(/api key/i)).toBeNull();
    }
  });
});

describe('Settings: editing', () => {
  it('Save sends only the changed fields; then refetches and invalidates capabilities, health and session', async () => {
    const { put } = mockApi();
    const { invalidate } = renderSettings();
    const assistant = await screen.findByTestId('settings-section-assistant');
    expect(screen.getByTestId('settings-savebar').dataset.dirty).toBe('false');
    expect(screen.getByRole('button', { name: 'Save' })).toHaveProperty('disabled', true);

    fireEvent.click(within(assistant).getByRole('button', { name: 'Move Mistral up' }));
    tab('Access');
    const limit = within(section('access')).getByRole('spinbutton', { name: 'Guest analyses per hour' });
    fireEvent.change(limit, { target: { value: '7' } });
    // A change made and undone is not sent.
    fireEvent.change(within(section('access')).getByRole('spinbutton', { name: 'Guest session hours' }), { target: { value: '12' } });
    fireEvent.change(within(section('access')).getByRole('spinbutton', { name: 'Guest session hours' }), { target: { value: '24' } });

    const bar = screen.getByTestId('settings-savebar');
    expect(bar.dataset.dirty).toBe('true');
    expect(bar.textContent).toContain('Unsaved changes: 2 fields in Assistant, Access');
    expect(screen.getByRole('button', { name: /^Assistant/ }).querySelector('[aria-label="unsaved changes"]')).toBeTruthy();

    save();
    await waitFor(() => expect(put).toHaveBeenCalledTimes(1));
    expect(put).toHaveBeenCalledWith('/v1/settings', {
      assistant: { provider_order: ['mistral', 'ollama'] },
      access: { guest_analyses_per_hour: 7 },
    });
    await waitFor(() => expect(screen.getByTestId('settings-savebar').textContent).toContain('Saved.'));
    const keys = invalidate.mock.calls.map(([filters]) => JSON.stringify(filters?.queryKey));
    for (const key of [['settings'], ['capabilities'], ['health'], ['session']]) expect(keys).toContain(JSON.stringify(key));
  });

  it('Discard drops the draft', async () => {
    const { put } = mockApi();
    renderSettings();
    await screen.findByTestId('settings-section-assistant');
    tab('Access');
    fireEvent.change(within(section('access')).getByRole('spinbutton', { name: 'Guest analyses per hour' }), { target: { value: '7' } });
    fireEvent.click(screen.getByRole('button', { name: 'Discard' }));
    expect(screen.getByTestId('settings-savebar').dataset.dirty).toBe('false');
    expect(within(section('access')).getByRole('spinbutton', { name: 'Guest analyses per hour' })).toHaveProperty('value', '1000');
    expect(put).not.toHaveBeenCalled();
  });

  it('chips: add and remove emails and domains', async () => {
    const { put } = mockApi();
    renderSettings();
    await screen.findByTestId('settings-section-assistant');
    tab('Access');
    const access = section('access');
    const add = within(access).getByRole('textbox', { name: 'Add to Allowed domains' });
    fireEvent.change(add, { target: { value: 'UNIGE.ch, umd.edu' } });
    fireEvent.keyDown(add, { key: 'Enter' });
    expect(within(within(access).getByRole('list', { name: 'Allowed domains' })).getAllByTestId('chip').map((c) => c.textContent)).toEqual([
      'unige.ch',
      'umd.edu',
    ]);
    fireEvent.click(within(access).getByRole('button', { name: 'Remove umd.edu' }));
    save();
    await waitFor(() => expect(put).toHaveBeenCalledWith('/v1/settings', { access: { allowed_domains: ['unige.ch'] } }));
  });

  it('a 422 shows inline under its field, the self-lockout guard included', async () => {
    const settings = doc();
    settings.access.operator_emails = ['oscar@operator.e2e'];
    const { put } = mockApi(settings);
    renderSettings();
    await screen.findByTestId('settings-section-assistant');
    tab('Access');
    fireEvent.click(within(section('access')).getByRole('button', { name: 'Remove oscar@operator.e2e' }));
    // Leave the section: the error brings it back.
    tab('Server');
    put.mockRejectedValueOnce(
      new ApiError('Keep your own address', 422, {
        detail: [
          {
            type: 'value_error',
            loc: ['body', 'access', 'operator_emails'],
            msg: 'Keep your own address (oscar@operator.e2e): without it you lose operator access',
          },
        ],
      }),
    );
    save();
    await waitFor(() => expect(screen.getByTestId('settings-section-access')).toBeTruthy());
    const error = within(field('access.operator_emails')).getByTestId('field-error');
    expect(error.textContent).toBe('Keep your own address (oscar@operator.e2e): without it you lose operator access');
    const input = within(field('access.operator_emails')).getByRole('textbox', { name: 'Add to Operator emails' });
    expect(input.getAttribute('aria-describedby')).toBe(error.id);
    expect(screen.getByTestId('settings-savebar').dataset.dirty).toBe('true');
    expect(screen.getByRole('button', { name: /^Access/ }).querySelector('[aria-label="has errors"]')).toBeTruthy();
    expect(put).toHaveBeenCalledWith('/v1/settings', { access: { operator_emails: [] } });
  });

  it('Reset to .env is offered for an override and sends null', async () => {
    const settings = doc();
    settings.access.source!.guest_analyses_per_hour = 'override';
    settings.access.guest_analyses_per_hour = 9;
    const { put } = mockApi(settings);
    renderSettings();
    await screen.findByTestId('settings-section-assistant');
    tab('Access');
    expect(within(field('access.guest_session_hours')).queryByRole('button', { name: 'Reset to .env' })).toBeNull();
    const row = field('access.guest_analyses_per_hour');
    expect(within(row).getByTestId('setting-source').textContent).toBe('override');
    fireEvent.click(within(row).getByRole('button', { name: 'Reset to .env' }));
    expect(row.textContent).toContain('Goes back to the .env value when you save.');
    expect(within(row).getByRole('spinbutton')).toHaveProperty('disabled', true);
    save();
    await waitFor(() => expect(put).toHaveBeenCalledWith('/v1/settings', { access: { guest_analyses_per_hour: null } }));
  });

  it('Test probes the providers and shows ok and latency', async () => {
    const { post } = mockApi();
    renderSettings();
    const assistant = await screen.findByTestId('settings-section-assistant');
    fireEvent.click(within(assistant).getByRole('button', { name: 'Test' }));
    const probe = await within(assistant).findByTestId('probe');
    expect(post).toHaveBeenCalledWith('/v1/settings/assistant/test', undefined, { timeout: 60_000 });
    expect(probe.dataset.ok).toBe('true');
    expect(probe.textContent).toContain('Ollama');
    expect(probe.textContent).toContain('OK · 4 ms');
  });

  it('Pre-download starts the download and follows its progress', async () => {
    const { post, get } = mockApi();
    renderSettings();
    await screen.findByTestId('settings-section-assistant');
    tab('Models');
    fireEvent.click(screen.getByRole('button', { name: 'Pre-download Prithvi EO v1 (100M)' }));
    await waitFor(() => expect(post).toHaveBeenCalledWith('/v1/ml-models/prithvi-eo-v1-100m/download'));
    const row = screen.getAllByTestId('settings-model')[0];
    await waitFor(() => expect(row.textContent).toContain('Downloading · 42 %'));
    expect(get).toHaveBeenCalledWith('/v1/ml-models/prithvi-eo-v1-100m');
  });

  it('a refused pre-download says why on the row', async () => {
    const { post } = mockApi();
    post.mockRejectedValueOnce(new ApiError('Nobody may download models on this server.', 403));
    renderSettings();
    await screen.findByTestId('settings-section-assistant');
    tab('Models');
    fireEvent.click(screen.getByRole('button', { name: 'Pre-download Prithvi EO v1 (100M)' }));
    expect((await screen.findByRole('alert')).textContent).toBe('Nobody may download models on this server.');
  });
});
