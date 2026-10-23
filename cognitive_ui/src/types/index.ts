import type { DrawnRegion } from '../store/useAppStore';

// Chat (one store: src/store/useChatStore.ts). One assistant message narrates
// one analysis (design/DESIGN.md §4), so a job is a single `analysis` message
// whose status and timeline are updated in place.

/** A file reference kept on a sent user message: metadata and URLs, never image bytes. */
export interface MessageAttachment {
  id: string;
  filename: string;
  path?: string;  // Server path for follow-up requests
  preview_url?: string;  // Server-relative preview URL from POST /v1/upload
}

interface BaseMessage {
  id: string;
  createdAt: string;  // ISO 8601
}

export interface UserMessage extends BaseMessage {
  kind: 'user';
  text: string;
  attachments?: MessageAttachment[];
  region?: DrawnRegion;
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
}

export interface ExecutionResult {
  ok: boolean;
  plan?: Plan;
  result?: any;
  progress?: any[];
  error?: string;
  candidate?: any;
}
