export type Role = 'user' | 'assistant' | 'system';

// Message types for chat
export type ChatMessageType = 'text' | 'job_submitted' | 'job_result' | 'error';

// Base chat message
export interface ChatMessage {
  role: Role;
  content: string;
  type?: ChatMessageType;
  jobId?: string;
  timestamp?: string;
  // For job_result messages: includes the parsed result data for display
  resultData?: JobResultData;
  // For user messages with attachments: includes preview images and path for context
  attachments?: Array<{
    id: string;
    filename: string;
    preview_png_base64?: string;
    path?: string;  // Server path for follow-up requests
  }>;
}

// Chat session for persistence
export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  jobIds: string[]; // All jobs associated with this session
  createdAt: string;
  updatedAt: string;
}

/** Extended message with job result data for display in chat. */
export interface JobResultMessage extends ChatMessage {
  type: 'job_result';
  jobId: string;
  jobStatus: JobStatus;
  result?: JobResultData;
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

  // Visualizations (base64 images, NOT included in LLM context)
  visualizations?: {
    type: string;
    label: string;
    base64: string;
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

// Context item for LLM processing (images stripped out)
export interface ContextItem {
  role: Role;
  content: string;
  // Metadata for context management
  tokenEstimate?: number;
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
  // Preview image for display in chat
  preview_png_base64?: string;
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
