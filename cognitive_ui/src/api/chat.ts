import { API_BASE_URL } from './client';
import { readSSE } from './sse';
import type { Attachment, ChatRegion, JobEvent, PendingIntent, RegionDataset } from '../types';

// POST /v1/chat, the Workspace's single entry point (docs/API.md "Conversation").

export interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
}

/** An attachment as the server's `Attachment` model takes it (`mime_type` is required). */
export interface ChatAttachment {
  id: string;
  filename: string;
  path: string;
  mime_type: string;
  size_bytes?: number;
  metadata?: Attachment['metadata'];
}

/** `JobRegion`: the server forbids extra fields, so only these. */
export interface ChatRegionRequest {
  geometry: GeoJSON.Geometry;
  name?: string;
  place?: string;
  dataset: RegionDataset;
  start_date: string;
  end_date: string;
  cloud_max: number;
}

export interface ChatRequest {
  message: string;
  history: ChatTurn[];
  attachments: ChatAttachment[];
  region?: ChatRegionRequest;
  pending?: PendingIntent;
}

/** The job as `analysis` carries it; F3 reads the rest. */
export interface ChatJob {
  id: string;
  status: string;
  events?: JobEvent[];  // stored as {seq, type, data}, the same list GET /v1/jobs/{id} returns
  region?: { line?: string; area_km2?: number } | null;
  [key: string]: unknown;
}

export type ChatStreamEvent =
  | { type: 'reply_delta'; text: string }
  | { type: 'reply_done'; text: string; pending?: PendingIntent }
  | { type: 'analysis'; job_id: string; job: ChatJob };

/** A refusal before any stream starts: 429 `GUEST_QUOTA`, 422, or another HTTP error. */
export class ChatError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = 'ChatError';
  }
}

export function toChatAttachment(a: Attachment): ChatAttachment {
  return {
    id: a.id,
    filename: a.filename,
    path: a.path,
    mime_type: a.mime_type ?? 'image/tiff',
    size_bytes: a.size_bytes,
    metadata: a.metadata,
  };
}

export function toChatRegion(r: ChatRegion): ChatRegionRequest {
  return {
    geometry: r.geometry,
    ...(r.name ? { name: r.name } : {}),
    ...(r.place ? { place: r.place } : {}),
    dataset: r.dataset,
    start_date: r.start_date,
    end_date: r.end_date,
    cloud_max: r.cloud_max,
  };
}

/** A refusal as `ChatError`: `{ok:false, error:{code, message}}` or FastAPI's `detail`. */
export async function errorFrom(res: Response): Promise<ChatError> {
  let body: { error?: { code?: string; message?: string; details?: Record<string, unknown> }; detail?: unknown } = {};
  try {
    body = await res.json();
  } catch {
    // not JSON: fall through to the status text
  }
  const detail = Array.isArray(body.detail)
    ? (body.detail[0] as { msg?: string } | undefined)?.msg
    : typeof body.detail === 'string'
      ? body.detail
      : undefined;
  const message = body.error?.message ?? detail ?? (res.statusText || `HTTP ${res.status}`);
  return new ChatError(message, res.status, body.error?.code, body.error?.details);
}

/**
 * Send one message as SSE and call `onEvent` per frame. Throws `ChatError` for a
 * refusal (before the stream) and rethrows network errors.
 */
export async function streamChat(
  request: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(request),
    signal,
  });
  if (!res.ok) throw await errorFrom(res);
  if (!res.body) throw new ChatError('Empty response from the server', res.status);

  for await (const frame of readSSE(res.body)) {
    if (frame.event !== 'reply_delta' && frame.event !== 'reply_done' && frame.event !== 'analysis') continue;
    onEvent({ type: frame.event, ...JSON.parse(frame.data) } as ChatStreamEvent);
  }
}
