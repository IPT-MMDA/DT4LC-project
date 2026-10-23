import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import type { AnalysisMessage, Job } from '../types';
import { findConversationByJobId, useChatStore } from '../store/useChatStore';
import { useJobSync } from './useJobSync';

const get = vi.hoisted(() => vi.fn());
vi.mock('../api/client', () => ({ default: { get } }));

const analysisOf = (jobId: string) =>
  findConversationByJobId(useChatStore.getState().conversations, jobId)!.messages.find(
    (m) => m.kind === 'analysis' && m.jobId === jobId,
  ) as AnalysisMessage;

beforeEach(() => {
  localStorage.clear();
  get.mockReset();
  useChatStore.setState({ conversations: [], activeConversationId: null });
});

describe('useJobSync', () => {
  it('settles unfinished analyses in every conversation on mount', async () => {
    const { addMessage, newConversation } = useChatStore.getState();
    addMessage({ kind: 'user', text: 'hello' });
    addMessage({ kind: 'analysis', jobId: 'job-a', status: 'pending', text: '', events: [] });
    newConversation();
    addMessage({ kind: 'user', text: 'ndvi' });
    addMessage({ kind: 'analysis', jobId: 'job-b', status: 'running', text: '', events: [] });
    addMessage({ kind: 'analysis', jobId: 'job-c', status: 'done', text: '', events: [] });

    get.mockImplementation(async (url: string): Promise<Job> => {
      if (url === '/v1/jobs/job-a') {
        return { id: 'job-a', status: 'completed', progress: 1, result: { intent: 'conversation', response: 'Hi!' } };
      }
      throw new Error('Request failed with status code 404');
    });

    renderHook(() => useJobSync());

    await waitFor(() => expect(analysisOf('job-b').status).toBe('failed'));
    expect(analysisOf('job-a')).toMatchObject({ status: 'done', result: { summary: 'Hi!', conversational: true } });
    expect(analysisOf('job-b').error).toMatch(/not found/i);
    expect(get).toHaveBeenCalledTimes(2); // job-c is already done
  });
});
