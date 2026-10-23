import { useQuery } from '@tanstack/react-query';
import apiClient from '../client';

/**
 * The parts of GET /v1/capabilities the Workspace reads (docs/API.md); the
 * Capabilities page (Phase 3) extends this type.
 */
export interface Capabilities {
  summary: {
    deployment: 'demo' | 'local';
    /** e.g. {deployment: "local server", assistant: "Gemini 3.5 Flash connected"}. */
    parts: { deployment: string; role: string; assistant: string; models: string; earth_engine: string };
    line: string;
  };
  /** `display` is the assistant's display name, e.g. "Gemini 3.5 Flash"; null offline. */
  assistant: { connected: boolean; provider: string | null; model: string | null; display: string | null };
  offline: boolean;
}

/** GET /v1/capabilities: what this server can do. It changes only with the server's setup. */
export function useCapabilities() {
  return useQuery({
    queryKey: ['capabilities'],
    queryFn: () => apiClient.get<Capabilities>('/v1/capabilities'),
    staleTime: 5 * 60_000,
  });
}
