import { API_BASE_URL } from './client';
import { errorFrom } from './chat';
import type { JobRecord } from './jobs';

// The Library's endpoints (BR2): job summaries with their stored `card`, deletes and
// the statistics CSV. fetch, not axios: a delete needs the HTTP status (409 for a
// running job), which the axios interceptor drops. GET /v1/files is useListFiles.

/** react-query key of GET /v1/jobs (the Library list); invalidate it after a delete. */
export const JOBS_QUERY_KEY = ['jobs'] as const;

/** GET /v1/jobs allows at most 100 per page. */
const JOBS_PAGE = 100;

/** A job's stored Library summary; null until the job finishes, and on rows from before BR2. */
export interface JobCard {
  title: string;
  analysis: { id: string; label: string } | null;
  kind: 'change' | 'index' | 'classification' | 'fields' | 'statistics' | 'composed' | 'image' | null;
  thumbnail_url: string | null;
  duration_s: number | null;
  error: { code: string; message: string } | null;
}

/** A GET /v1/jobs item: the job record without `result`, `events`, `proposal` and `region`. */
export interface JobSummary extends Pick<JobRecord, 'id' | 'status' | 'prompt'> {
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  card?: JobCard | null;
}

export async function listJobs(): Promise<JobSummary[]> {
  const res = await fetch(`${API_BASE_URL}/v1/jobs?limit=${JOBS_PAGE}`);
  if (!res.ok) throw await errorFrom(res);
  return ((await res.json()) as { jobs: JobSummary[] }).jobs;
}

async function remove(path: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}${path}`, { method: 'DELETE' });
  if (!res.ok) throw await errorFrom(res);
}

/** DELETE /v1/jobs/{id}. Throws `ChatError`: 409 while pending or running, 404 for an unknown job. */
export const deleteJob = (jobId: string) => remove(`/v1/jobs/${encodeURIComponent(jobId)}`);

/** DELETE /v1/files/{id}: an `upload_id` (the sample's deletes the pair) or a file stem. */
export const deleteFile = (fileId: string) => remove(`/v1/files/${encodeURIComponent(fileId)}`);

/** GET /v1/jobs/{id}/statistics.csv, as a download link. */
export const statisticsCsvUrl = (jobId: string) => `${API_BASE_URL}/v1/jobs/${encodeURIComponent(jobId)}/statistics.csv`;

/** GET /v1/download of a server path (a file's own `path`). */
export const downloadUrl = (path: string) => `${API_BASE_URL}/v1/download?path=${encodeURIComponent(path)}`;
