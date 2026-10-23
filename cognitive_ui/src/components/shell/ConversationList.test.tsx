import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../../api/client';
import { useConversationRetention } from '../../hooks/useConversationRetention';
import { useChatStore } from '../../store/useChatStore';
import { GUEST_SESSION, MEMBER_SESSION, OPERATOR_SESSION } from '../../test/fixtures';
import type { SessionInfo } from '../../types';
import { ConversationList } from './ConversationList';
import { emptyListText } from './emptyListText';

const assistant = (connected: boolean) => ({
  connected,
  offline: !connected,
  provider: connected ? 'gemini' : null,
  model: connected ? 'gemini-3.5-flash' : null,
});
const capabilities = (connected: boolean) => ({
  summary: {
    deployment: 'local',
    parts: { deployment: 'local server', role: 'member account', assistant: '', models: '', earth_engine: '' },
    line: '',
  },
  assistant: { ...assistant(connected), display: connected ? 'Gemini 3.5 Flash' : null },
  offline: !connected,
});

function mockApi(session: SessionInfo, connected = false) {
  vi.spyOn(apiClient, 'get').mockImplementation(async (url: string) => {
    if (url === '/v1/session') return session as never;
    if (url === '/v1/health') return { ok: true, service: 'DT4LC', version: '0.6.0', assistant: assistant(connected) } as never;
    if (url === '/v1/capabilities') return capabilities(connected) as never;
    throw new Error(`unexpected GET ${url}`);
  });
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const renderList = () => render(<ConversationList />, { wrapper });
const list = () => screen.getByRole('complementary', { name: 'Conversations' });
const titles = () => useChatStore.getState().conversations.map((c) => c.title);

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, keptHours: 24, pendingAttachments: [], draft: '' });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('the session badge', () => {
  it.each([
    ['guest, offline', GUEST_SESSION, false, 'Guest. 4 of 5 analyses left this hour · kept 24 h · no account.'],
    ['guest, online', GUEST_SESSION, true, 'Guest. 4 of 5 analyses left this hour · kept 24 h · no account.'],
    ['member, offline', MEMBER_SESSION, false, 'Member. Local server · no analysis limit · assistant offline.'],
    ['member, online', MEMBER_SESSION, true, 'Member. Local server · no analysis limit · Gemini 3.5 Flash connected.'],
    ['operator, offline', OPERATOR_SESSION, false, 'Operator. Local server · no analysis limit · assistant offline.'],
    ['operator, online', OPERATOR_SESSION, true, 'Operator. Local server · no analysis limit · Gemini 3.5 Flash connected.'],
  ])('%s', async (_name, session, connected, text) => {
    mockApi(session, connected);
    renderList();
    const badge = await screen.findByTestId('session-badge');
    await waitFor(() => expect(badge.textContent).toBe(text));
  });

  it('one analysis left: the quota part in the canc hue', async () => {
    mockApi({ ...GUEST_SESSION, analyses: { limit: 5, used: 4, remaining: 1, resets_at: GUEST_SESSION.analyses.resets_at } });
    renderList();
    const low = await screen.findByTestId('low-quota');
    expect(low.textContent).toBe('1 of 5 analyses left this hour');
    expect(low.className).toContain('text-canc-ink');
    expect(screen.getByTestId('session-badge').textContent).toBe('Guest. 1 of 5 analyses left this hour · kept 24 h · no account.');
  });

  it('more left: no hue', async () => {
    mockApi(GUEST_SESSION);
    renderList();
    await screen.findByTestId('session-badge');
    expect(screen.queryByTestId('low-quota')).toBeNull();
  });
});

describe('the empty list', () => {
  it('"No conversations yet" and how long they stay', () => {
    expect(emptyListText(24)).toBe('No conversations yet. They appear here as you work and stay for 24 h.');
    expect(emptyListText(6)).toBe('No conversations yet. They appear here as you work and stay for 6 h.');
    expect(emptyListText(null)).toBe('No conversations yet. They appear here as you work.');
  });

  it("follows the session's retention", async () => {
    mockApi(GUEST_SESSION);
    renderList();
    expect(screen.getByTestId('no-conversations').textContent).toBe(emptyListText(24));
    cleanup();

    mockApi(MEMBER_SESSION);
    renderHook(() => useConversationRetention(), { wrapper });
    await waitFor(() => expect(useChatStore.getState().keptHours).toBeNull());
    renderList();
    expect(screen.getByTestId('no-conversations').textContent).toBe(emptyListText(null));
  });

  it("a guest session's kept_hours reaches the store", async () => {
    useChatStore.setState({ keptHours: null });
    mockApi({ ...GUEST_SESSION, kept_hours: 6 });
    renderHook(() => useConversationRetention(), { wrapper });
    await waitFor(() => expect(useChatStore.getState().keptHours).toBe(6));
  });
});

describe('rename and delete from the list', () => {
  beforeEach(() => {
    mockApi(GUEST_SESSION);
    useChatStore.getState().addMessage({ kind: 'user', text: 'Older chat' });
    useChatStore.getState().newConversation();
    useChatStore.getState().addMessage({ kind: 'user', text: 'Water after the breach' });
  });

  it('Rename edits the title in place: Enter saves, Escape and an empty name keep the old one', () => {
    renderList();
    fireEvent.click(within(list()).getByRole('button', { name: 'Rename “Older chat”' }));
    const input = screen.getByLabelText('Conversation name') as HTMLInputElement;
    expect(input.value).toBe('Older chat');
    fireEvent.change(input, { target: { value: '  Kherson fields  ' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(titles()).toEqual(['Water after the breach', 'Kherson fields']);
    expect(screen.queryByLabelText('Conversation name')).toBeNull();

    fireEvent.click(within(list()).getByRole('button', { name: 'Rename “Kherson fields”' }));
    fireEvent.change(screen.getByLabelText('Conversation name'), { target: { value: 'Never mind' } });
    fireEvent.keyDown(screen.getByLabelText('Conversation name'), { key: 'Escape' });
    expect(titles()).toEqual(['Water after the breach', 'Kherson fields']);

    fireEvent.click(within(list()).getByRole('button', { name: 'Rename “Kherson fields”' }));
    fireEvent.change(screen.getByLabelText('Conversation name'), { target: { value: '   ' } });
    fireEvent.blur(screen.getByLabelText('Conversation name'));
    expect(titles()).toEqual(['Water after the breach', 'Kherson fields']);
  });

  it('a blur saves the new name', () => {
    renderList();
    fireEvent.click(within(list()).getByRole('button', { name: 'Rename “Older chat”' }));
    fireEvent.change(screen.getByLabelText('Conversation name'), { target: { value: 'Renamed' } });
    fireEvent.blur(screen.getByLabelText('Conversation name'));
    expect(titles()).toEqual(['Water after the breach', 'Renamed']);
  });

  it('Delete asks first; Cancel keeps it, Delete removes it and leaves the active one', () => {
    const active = useChatStore.getState().activeConversationId;
    renderList();
    fireEvent.click(within(list()).getByRole('button', { name: 'Delete “Older chat”' }));
    const confirm = screen.getByRole('group', { name: 'Delete this conversation?' });
    fireEvent.click(within(confirm).getByRole('button', { name: 'Cancel' }));
    expect(titles()).toEqual(['Water after the breach', 'Older chat']);
    expect(screen.queryByRole('group', { name: 'Delete this conversation?' })).toBeNull();

    fireEvent.click(within(list()).getByRole('button', { name: 'Delete “Older chat”' }));
    fireEvent.click(within(screen.getByRole('group', { name: 'Delete this conversation?' })).getByRole('button', { name: 'Delete' }));
    expect(titles()).toEqual(['Water after the breach']);
    expect(useChatStore.getState().activeConversationId).toBe(active);
  });

  it('deleting the active conversation goes back to a new one; the last one shows the empty list', () => {
    renderList();
    fireEvent.click(within(list()).getByRole('button', { name: 'Delete “Water after the breach”' }));
    fireEvent.click(within(screen.getByRole('group', { name: 'Delete this conversation?' })).getByRole('button', { name: 'Delete' }));
    expect(useChatStore.getState().activeConversationId).toBeNull();
    fireEvent.click(within(list()).getByRole('button', { name: 'Delete “Older chat”' }));
    fireEvent.click(within(screen.getByRole('group', { name: 'Delete this conversation?' })).getByRole('button', { name: 'Delete' }));
    expect(screen.getByTestId('no-conversations').textContent).toBe(emptyListText(24));
  });

  it('every control is a real <button>', () => {
    renderList();
    for (const b of within(list()).getAllByRole('button')) {
      expect(b.tagName).toBe('BUTTON');
      expect(b.getAttribute('type')).toBe('button');
    }
  });
});
