// Chat (one store: src/store/useChatStore.ts). One assistant message narrates
// one analysis (design/DESIGN.md §4), so a job is a single `analysis` message
// whose status and timeline are updated in place.

/** A file reference kept on a sent user message: metadata and URLs, never image bytes. */
export interface MessageAttachment {
  id: string;
  filename: string;
  path?: string;  // Server path for follow-up requests
  preview_url?: string;  // Server-relative preview URL from POST /v1/upload
  group?: AttachmentGroup;  // Shown as one chip with the other files of the group
  mime_type?: string;  // re-sent with follow-ups (the server requires it)
  size_bytes?: number;
  metadata?: UploadMetadata;  // re-sent with follow-ups (the intent names the date)
}

/**
 * Files shown as one chip, e.g. the two dates of the Kakhovka sample from
 * POST /v1/sample: `{name: "kakhovka_hls_l30", line: "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09"}`.
 */
export interface AttachmentGroup {
  name: string;
  line: string;
}

/** `metadata` of POST /v1/upload (docs/API.md); a best guess, any field may be missing. */
export interface UploadMetadata {
  sensor?: string;
  bands?: { index: number; code: string; description: string }[];
  dtype?: string;
  resolution_m?: number;
  date?: string;
  size_bytes?: number;
  size_label?: string;
  crs?: string;
  bounds_wgs84?: [number, number, number, number];
  line?: string;  // chip text without the size; "" when nothing could be read
}

interface BaseMessage {
  id: string;
  createdAt: string;  // ISO 8601
}

export interface UserMessage extends BaseMessage {
  kind: 'user';
  text: string;
  attachments?: MessageAttachment[];
  region?: ChatRegion;
}

export type RegionDataset = 'HLS L30' | 'HLS S30' | 'Sentinel-2' | 'Landsat 8/9' | 'MODIS';

/**
 * A region from the map as the composer holds it and POST /v1/chat sends it
 * (`region` of docs/API.md). `area_km2` is computed in the browser until the
 * job echoes the region; `line` is the server's echo, which wins when present.
 */
export interface ChatRegion {
  id: string;  // the map store's DrawnRegion id
  geometry: GeoJSON.Geometry;
  name?: string;
  place?: string;
  dataset: RegionDataset;
  start_date: string;  // YYYY-MM-DD
  end_date: string;
  cloud_max: number;
  area_km2: number;
  line?: string;
}

/** `pending` of a reply that asked for a file; sent back with the next message. */
export interface PendingIntent {
  prompt: string;
  item_id: string | null;
  needs: string[];
  asked_at: string;
}

/** A conversational reply (or a submit error) that is not tied to a job. */
export interface AssistantMessage extends BaseMessage {
  kind: 'assistant';
  text: string;
  error?: boolean;
}

export type AnalysisStatus = 'pending' | 'running' | 'done' | 'failed' | 'cancelled';

/** One entry of the analysis timeline, as streamed by GET /v1/jobs/{id}/events. */
export interface AnalysisEvent {
  type: string;  // intent, proposal, planning, step_started, step_progress, step_finished, done, failed, cancelled
  [key: string]: unknown;
}

/** A job event as the server stores it: `events` of a JobRecord and of the `analysis` chat frame. */
export interface JobEvent extends AnalysisEvent {
  seq: number;
  data: Record<string, unknown>;
}

export interface AnalysisMessage extends BaseMessage {
  kind: 'analysis';
  jobId: string;
  status: AnalysisStatus;
  text: string;  // narration (intent); empty until the assistant says something
  events: AnalysisEvent[];  // the timeline slot
  result?: JobResultData;
  error?: string;
}

export type ChatMessage = UserMessage | AssistantMessage | AnalysisMessage;

/** A message as callers add it; the store assigns `id` and `createdAt` when missing. */
export type NewChatMessage =
  | (Omit<UserMessage, 'id' | 'createdAt'> & Partial<BaseMessage>)
  | (Omit<AssistantMessage, 'id' | 'createdAt'> & Partial<BaseMessage>)
  | (Omit<AnalysisMessage, 'id' | 'createdAt'> & Partial<BaseMessage>);

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  pending?: PendingIntent | null;  // the latest reply's pending; a reply without one clears it
}

// Structured result data for context management
export interface JobResultData {
  // Text summary for LLM context (kept small)
  summary?: string;

  // Flag for conversational responses (no pipeline execution)
  conversational?: boolean;

  // Statistics (numbers only, suitable for context)
  statistics?: {
    type: 'ndvi' | 'ndwi' | 'ndsi' | 'lulc' | 'snow' | 'change' | 'statistics' | 'features' | 'field_boundaries';
    values: Record<string, number | string>;
  };

  // Visualizations (server-relative PNG URLs, NOT included in LLM context)
  visualizations?: {
    type: string;
    label: string;
    url: string;
  }[];

  // Classification data
  classification?: Record<string, { pixels: number; percentage: number }>;

  // Field boundaries (for Delineate-Anything)
  fieldBoundaries?: {
    numFields: number;
    totalAreaM2: number;
    outputPath: string;
    crs: string;
  };

  // Reconstruction (for Prithvi MAE)
  reconstruction?: {
    model: string;
    inputFile: string;
    outputDir: string;
  };
}

export interface Plan {
  tags: string[];
  goals: string[];
  pipeline: string[];
  inputs: Record<string, any>;
  meta: Record<string, any>;
}

/** Job status values matching backend JobStatus enum. */
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/** Job model matching backend Job.to_dict() response. */
export interface Job {
  id: string;
  status: JobStatus;
  progress: number;
  message?: string;
  result?: Record<string, any>;
  plan?: Plan;
  error?: string;
  created_at?: string;
  updated_at?: string;
}

export interface Attachment {
  id: string;
  filename: string;
  path: string;
  mime_type?: string;
  size_bytes?: number;
  // Server-relative preview URL for display in chat (never the image bytes)
  preview_url?: string;
  metadata?: UploadMetadata;  // sent back with jobs and chat messages
  group?: AttachmentGroup;
}

export interface JobSubmitRequest {
  prompt: string;
  mode?: 'hybrid' | 'llm' | 'template';
  attachments?: Attachment[];
  context?: Record<string, any>;
}

export interface JobsListResponse {
  jobs: Job[];
  total: number;
  limit: number;
  offset: number;
}

export interface HealthResponse {
  ok: boolean;
  service: string;
  version: string;
  /** Always present since v0.6.0; `offline` is `!connected`. */
  assistant?: {
    connected: boolean;
    offline: boolean;
    provider: string | null;
    model: string | null;
  };
}

/** GET /v1/session: the caller's role, guest quota and retention (docs/API.md). */
export interface SessionInfo {
  role: 'guest' | 'member' | 'operator';
  guest: boolean;
  analyses: {
    limit: number | null;  // null: no limit (members, operators)
    used: number;
    remaining: number | null;
    resets_at: string | null;
  };
  kept_hours: number | null;
  expires_at: string;
  can_download_models: boolean;
}

export interface ExecutionResult {
  ok: boolean;
  plan?: Plan;
  result?: any;
  progress?: any[];
  error?: string;
  candidate?: any;
}
