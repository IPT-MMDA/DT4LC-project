import apiClient from './client';

/**
 * Operator settings: GET/PUT /v1/settings and POST /v1/settings/assistant/test
 * (server/routes/settings.py). Both 404 for guests and members. Keys are only ever
 * reported as `key_present`; nothing here sends or receives a key value.
 */
export const SETTINGS_QUERY_KEY = ['settings'] as const;

/** Where a writable value comes from: the operator's overrides file, `.env`, or the built-in default. */
export type SettingSource = 'override' | 'env' | 'default';

/** Every section and provider: `editable[field]` says whether PUT takes it; `source` is set for editable fields. */
interface Editable {
  editable: Record<string, boolean>;
  source?: Record<string, SettingSource>;
}

export interface SettingsProvider extends Editable {
  enabled: boolean;
  models: string[];
  multiple_models: boolean;
  key_required: boolean;
  key_present: boolean;
  available: boolean;
}

export interface SettingsAssistant extends Editable {
  provider_order: string[];
  providers: Record<string, SettingsProvider>;
  connected: boolean;
  provider: string | null;
  model: string | null;
  test_path: string;
}

export interface SettingsModelItem {
  id: string;
  display_name: string;
  installed: boolean;
  size_bytes: number;
  est_download_s: number;
  licence: string;
  licence_url: string;
  steps: string[];
}

export type DownloadPolicy = 'all' | 'members' | 'operators' | 'none';

export interface SettingsModels extends Editable {
  items: SettingsModelItem[];
  download_policy: DownloadPolicy;
  download_policy_options: DownloadPolicy[];
  pre_download: { method: string; path: string };
}

export interface SettingsAccess extends Editable {
  allowed_emails: string[];
  allowed_domains: string[];
  operator_emails: string[];
  guest_analyses_per_hour: number;
  guest_session_hours: number;
  bounds: Record<string, { min: number; max: number }>;
  sign_in: { configured: boolean };
}

export interface SettingsDataSources extends Editable {
  earth_engine: { configured: boolean; project_id_present: boolean; key_file_present: boolean };
  uploads: { path: string; files: number; bytes: number };
  sample: { name: string; bundled: boolean; path: string };
}

export interface SettingsStorage {
  bytes: number;
  files: number;
  truncated: boolean;  // the walk stopped early: bytes and files are lower bounds
}

export interface SettingsServer extends Editable {
  version: string;
  deployment: 'demo' | 'local';
  paths: Record<string, string>;
  storage: Record<string, SettingsStorage>;
  jobs: { total: number; by_status: Record<string, number>; queue_size: number; workers: number };
}

export interface SettingsDoc {
  assistant: SettingsAssistant;
  models: SettingsModels;
  access: SettingsAccess;
  data_sources: SettingsDataSources;
  server: SettingsServer;
}

/** PUT /v1/settings: any subset of the writable fields; `null` drops a field's override. */
export interface SettingsUpdate {
  assistant?: {
    provider_order?: string[] | null;
    providers?: Record<string, { enabled?: boolean | null; models?: string[] | null }>;
  };
  models?: { download_policy?: DownloadPolicy | null };
  access?: {
    allowed_emails?: string[] | null;
    allowed_domains?: string[] | null;
    operator_emails?: string[] | null;
    guest_analyses_per_hour?: number | null;
    guest_session_hours?: number | null;
  };
}

export interface SettingsProbe {
  name: string;
  model: string;
  ok: boolean;
  latency_ms: number;
  error: string | null;
}

export interface SettingsTestResult {
  connected: boolean;
  error?: string | null;
  providers: SettingsProbe[];
}

/** GET /v1/ml-models/{id}: `status` and, while downloading, `download_progress.percent`. */
export interface ModelDownloadStatus {
  id: string;
  status: 'not_installed' | 'downloading' | 'available' | 'failed';
  download_progress?: { percent: number };
  error?: string;
}

export const getSettings = () => apiClient.get<SettingsDoc>('/v1/settings');

export const putSettings = (update: SettingsUpdate) => apiClient.put<SettingsUpdate, SettingsDoc>('/v1/settings', update);

/** Probes every provider of the router; each may take up to 20 s on the server. */
export const testAssistant = (path: string) => apiClient.post<undefined, SettingsTestResult>(path, undefined, { timeout: 60_000 });

/** POST `pre_download.path` with `{id}` filled in: starts a background download (202). */
export const preDownloadModel = (path: string, id: string) =>
  apiClient.post<undefined, unknown>(path.replace('{id}', encodeURIComponent(id)));

export const getModelStatus = (id: string) => apiClient.get<ModelDownloadStatus>(`/v1/ml-models/${encodeURIComponent(id)}`);
