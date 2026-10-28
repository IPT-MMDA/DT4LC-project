import { useEffect } from 'react';
import { ChatError } from '../api/chat';
import { getJob, streamJobEvents, type JobRecord } from '../api/jobs';
import {
  analysisReducer,
  isTerminal,
  lastSeq,
  mergeEvents,
  statusOf,
} from '../components/chat/analysis/analysisReducer';
import { useChatStore } from '../store/useChatStore';
import type { AnalysisEvent, AnalysisMessage } from '../types';
import { apiLogger as logger } from '../utils/logger';

// The live side of an analysis message: GET /v1/jobs/{id}/events into the chat
// store. One stream per job however many components follow it; it closes on a
// terminal event, or when the last follower unmounts (a conversation switch), and
// a follower that mounts again resumes with `?after=<last seq>`.

const RETRY_MS = [500, 1000, 2000, 5000, 10_000];
const OFFLINE_SUMMARY = /summarization failed/i;

function findAnalysis(jobId: string): AnalysisMessage | undefined {
  for (const c of useChatStore.getState().conversations) {
    const m = c.messages.find((x) => x.kind === 'analysis' && x.jobId === jobId);
    if (m) return m as AnalysisMessage;
  }
  return undefined;
}

/**
 * Merge `events` into the job's message and keep its coarse status in step. The
 * step into `done` fetches the result, whether or not a stream is still open.
 */
export function applyJobEvents(jobId: string, events: AnalysisEvent[]): void {
  const message = findAnalysis(jobId);
  if (!message || events.length === 0) return;
  const before = analysisReducer(message.events).phase;
  const merged = mergeEvents(message.events, events);
  const view = analysisReducer(merged);
  useChatStore.getState().updateAnalysis(jobId, { events: merged, status: statusOf(view) });
  if (view.phase === 'done' && before !== 'done') void finish(jobId);
}

/**
 * The interpretation paragraph of a finished job: the agent-analysis step's text
 * (`result.execution.artifacts.agent_result.summary`). Without an assistant the step
 * reports "…summarization failed…", which is not an interpretation: none then.
 */
export function interpretationOf(job: Pick<JobRecord, 'result'>): string | undefined {
  const execution = (job.result?.execution ?? {}) as { artifacts?: Record<string, unknown> };
  const agent = execution.artifacts?.agent_result as { summary?: unknown } | undefined;
  const text = typeof agent?.summary === 'string' ? agent.summary.trim() : '';
  return text && !OFFLINE_SUMMARY.test(text) ? text : undefined;
}

/** After `done`: the job record's events (the reload source) and its interpretation. */
async function finish(jobId: string): Promise<void> {
  try {
    const job = await getJob(jobId);
    applyJobEvents(jobId, job.events ?? []);
    const summary = interpretationOf(job);
    if (summary) {
      const message = findAnalysis(jobId);
      useChatStore.getState().updateAnalysis(jobId, { result: { ...message?.result, summary } });
    }
  } catch (err) {
    logger.warn(`Could not read the result of job ${jobId}:`, err);
  }
}

const wait = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });

async function follow(jobId: string, signal: AbortSignal): Promise<void> {
  for (let attempt = 0; !signal.aborted; ) {
    const message = findAnalysis(jobId);
    if (!message) return;
    if (isTerminal(analysisReducer(message.events))) return;
    try {
      await streamJobEvents(jobId, lastSeq(message.events), (event) => {
        attempt = 0;
        applyJobEvents(jobId, [event]);
      }, signal);
    } catch (err) {
      if (signal.aborted) return;
      if (err instanceof ChatError && err.status === 404) {
        // Gone: a guest's job past its 24 h, or another session's.
        useChatStore.getState().updateAnalysis(jobId, { error: 'This analysis is no longer on the server.' });
        return;
      }
      logger.warn(`Event stream of job ${jobId} dropped:`, err);
      // The record carries the same events; it fills a gap the stream left.
      await getJob(jobId).then((job) => applyJobEvents(jobId, job.events ?? []), () => {});
    }
    if (signal.aborted || isTerminal(analysisReducer(findAnalysis(jobId)?.events ?? []))) return;
    await wait(RETRY_MS[Math.min(attempt++, RETRY_MS.length - 1)], signal);
  }
}

interface Subscription {
  followers: number;
  controller: AbortController;
}
const subscriptions = new Map<string, Subscription>();

/** Follow job `jobId` until it ends; call the returned function to stop following. */
export function subscribeJobEvents(jobId: string): () => void {
  let sub = subscriptions.get(jobId);
  if (sub) {
    sub.followers += 1;
  } else {
    const controller = new AbortController();
    sub = { followers: 1, controller };
    subscriptions.set(jobId, sub);
    void follow(jobId, controller.signal).finally(() => {
      if (subscriptions.get(jobId)?.controller === controller) subscriptions.delete(jobId);
    });
  }
  const mine = sub;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    mine.followers -= 1;
    if (mine.followers === 0) {
      mine.controller.abort();
      if (subscriptions.get(jobId) === mine) subscriptions.delete(jobId);
    }
  };
}

/** Jobs with an open stream; for tests. */
export const activeJobStreams = () => [...subscriptions.keys()];

/** Follow the analysis while it is mounted and not finished. */
export function useJobEvents(jobId: string, terminal: boolean): void {
  useEffect(() => {
    if (terminal) return;
    return subscribeJobEvents(jobId);
  }, [jobId, terminal]);
}
