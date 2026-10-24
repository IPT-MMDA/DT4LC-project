import { API_BASE_URL } from './client';
import { ChatError, errorFrom } from './chat';
import { readSSE } from './sse';
import type { JobEvent } from '../types';

// The job endpoints an analysis message calls (docs/API.md "Job Management").
// fetch, not axios: callers need the HTTP status and `error.code` of a refusal
// (409, 422, 429 GUEST_QUOTA), which the axios interceptor drops.

/** GET /v1/jobs/{id}, as far as the Workspace reads it. */
export interface JobRecord {
  id: string;
  status: 'pending' | 'proposal' | 'running' | 'completed' | 'failed' | 'cancelled';
  prompt?: string;
  events?: JobEvent[];
  result?: Record<string, unknown> | null;
  proposal?: { text: string; plan: unknown } | null;
  [key: string]: unknown;
}

async function send(method: 'GET' | 'POST', path: string, body?: unknown): Promise<JobRecord> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as JobRecord;
}

export const getJob = (jobId: string) => send('GET', `/v1/jobs/${encodeURIComponent(jobId)}`);

/** Run a proposal; `plan` is an edited plan ("Change"). Throws `ChatError` for 404, 409, 422 and 429. */
export const confirmJob = (jobId: string, plan?: unknown) =>
  send('POST', `/v1/jobs/${encodeURIComponent(jobId)}/confirm`, plan === undefined ? undefined : { plan });

export const cancelJob = (jobId: string) => send('POST', `/v1/jobs/${encodeURIComponent(jobId)}/cancel`);

/**
 * Follow GET /v1/jobs/{id}/events from `after` (0: the whole log) until the server
 * closes the stream (after a terminal event) or `signal` aborts. Each frame is passed
 * to `onEvent` as the stored `{seq, type, data}`. Throws `ChatError` for a refusal
 * (404: unknown job, or another session's) and rethrows network errors.
 */
export async function streamJobEvents(
  jobId: string,
  after: number,
  onEvent: (event: JobEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const query = after > 0 ? `?after=${after}` : '';
  const res = await fetch(`${API_BASE_URL}/v1/jobs/${encodeURIComponent(jobId)}/events${query}`, {
    headers: { Accept: 'text/event-stream' },
    signal,
  });
  if (!res.ok) throw await errorFrom(res);
  if (!res.body) throw new ChatError('Empty response from the server', res.status);
  for await (const frame of readSSE(res.body)) {
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(frame.data) as Record<string, unknown>;
    } catch {
      continue;
    }
    const seq = Number(frame.id);
    onEvent({ seq: Number.isFinite(seq) ? seq : 0, type: frame.event, data });
  }
}
