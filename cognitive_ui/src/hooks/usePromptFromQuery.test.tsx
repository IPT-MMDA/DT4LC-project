import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../api/client';
import { useChatStore } from '../store/useChatStore';
import { SAMPLE } from '../test/fixtures';
import type { Attachment } from '../types';
import { promptHref, usePromptFromQuery } from './usePromptFromQuery';
import { useSampleFromQuery } from './useSample';

const PROMPT = 'Compare water before and after the dam breach';
const ATTACHED: Attachment = { id: 'up1', filename: 'mine.tif', path: '/u/up1_mine.tif' };

// As in the Workspace: both query hooks (?sample= does nothing unless it is 1).
function renderAt(url: string) {
  const location = { current: '' };
  const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter initialEntries={[url]}>{children}</MemoryRouter>;
  renderHook(
    () => {
      useSampleFromQuery();
      usePromptFromQuery();
      const loc = useLocation();
      location.current = loc.pathname + loc.search;
    },
    { wrapper },
  );
  return location;
}

beforeEach(() => {
  localStorage.clear();
  useChatStore.setState({ conversations: [], activeConversationId: null, pendingAttachments: [], draft: '' });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('promptHref', () => {
  it('encodes the prompt as the Workspace query', () => {
    expect(promptHref('Where is open water on 9 June?')).toBe('/app?prompt=Where+is+open+water+on+9+June%3F');
    expect(new URLSearchParams(promptHref(PROMPT).split('?')[1]).get('prompt')).toBe(PROMPT);
  });
});

describe('?prompt=', () => {
  it('puts the prompt in the composer without sending, and drops the parameter', async () => {
    const post = vi.spyOn(apiClient, 'post');
    const location = renderAt(`${promptHref(PROMPT)}&x=2`);

    await waitFor(() => expect(location.current).toBe('/app?x=2'));
    const state = useChatStore.getState();
    expect(state.draft).toBe(PROMPT);
    expect(state.conversations).toEqual([]);
    expect(post).not.toHaveBeenCalled();
  });

  it('keeps the active conversation and the files already attached, and attaches nothing', async () => {
    const post = vi.spyOn(apiClient, 'post');
    useChatStore.setState({ activeConversationId: 'chat_old', pendingAttachments: [ATTACHED], draft: 'old text' });
    const location = renderAt(promptHref(PROMPT));

    await waitFor(() => expect(location.current).toBe('/app'));
    const state = useChatStore.getState();
    expect(state.draft).toBe(PROMPT);
    expect(state.activeConversationId).toBe('chat_old');
    expect(state.pendingAttachments).toEqual([ATTACHED]);
    expect(post).not.toHaveBeenCalled();
  });

  it('leaves the draft alone without the parameter', async () => {
    useChatStore.setState({ draft: 'typing…' });
    renderAt('/app?sample=0');
    await act(async () => {});
    expect(useChatStore.getState().draft).toBe('typing…');
  });

  it('works next to ?sample=1: the sample in a new conversation, the prompt in the composer', async () => {
    vi.spyOn(apiClient, 'post').mockImplementation(() => Promise.resolve(SAMPLE) as Promise<never>);
    useChatStore.setState({ activeConversationId: 'chat_old' });
    const location = renderAt(`/app?sample=1&${promptHref(PROMPT).split('?')[1]}`);

    await waitFor(() => expect(location.current).toBe('/app'));
    const state = useChatStore.getState();
    expect(state.draft).toBe(PROMPT);
    expect(state.pendingAttachments).toHaveLength(2);
    expect(state.activeConversationId).toBeNull();
    expect(state.conversations).toEqual([]);
  });
});
