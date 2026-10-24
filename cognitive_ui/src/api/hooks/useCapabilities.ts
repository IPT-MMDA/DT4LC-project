import { useQuery } from '@tanstack/react-query';
import type { SessionInfo } from '../../types';
import apiClient from '../client';

/** Model install state for the caller's role (docs/API.md); a built-in analysis that needs the model inherits it. */
export type ModelStatus = 'installed' | 'downloads_on_first_use' | 'not_available_for_guests' | 'not_installed';

export interface AskItem {
  id: string;
  prompt: string;
}

export interface ComposedItem {
  id: string;
  title: string;
  description: string;
  /** Mono line, e.g. "NDMI = (B5 − B6)/(B5 + B6) · NBR · SAVI · NDBI · custom". */
  recipe: string;
  badge: string;
  needs_assistant: boolean;
  needs_label: string;
  prompt: string;
}

export interface BuiltinItem {
  id: string;
  /** One of the tier's `groups` ids. */
  group: string;
  title: string;
  formula: string | null;
  description: string;
  inputs: string[];
  outputs: string[];
  time_estimate: string;
  runs_without_assistant: boolean;
  prompt: string;
  model: { id: string; title: string; short: string | null } | null;
  /** `available` without a model; with one, `installed` or the model's status. */
  status: 'available' | ModelStatus;
  status_label: string;
}

export interface ModelItem {
  id: string;
  title: string;
  description: string | null;
  size_bytes: number;
  size_label: string;
  download_time: string;
  licence: string;
  licence_url: string;
  used_by: string[];
  status: ModelStatus;
  status_label: string;
}

export interface DataItem {
  id: string;
  title: string;
  detail: string;
  status: 'always' | 'not_bundled' | 'configured' | 'not_configured';
  status_label: string;
}

export interface ComingItem {
  id: string;
  title: string;
  description: string | null;
}

interface TierBase<Id extends string, Item> {
  id: Id;
  number: number;
  title: string;
  subtitle: string | null;
  /** `connected`/`offline` (ask), `available`/`offline` (composed), `available`, or `coming`. */
  status: 'connected' | 'offline' | 'available' | 'coming';
  status_label: string | null;
  needs_assistant: boolean;
  badge: string | null;
  collapsed: boolean;
  groups: { id: string; title: string }[] | null;
  note: string | null;
  items: Item[];
}

export type AskTier = TierBase<'ask', AskItem>;
export type ComposedTier = TierBase<'composed', ComposedItem>;
export type BuiltinTier = TierBase<'builtin', BuiltinItem>;
export type ModelsTier = TierBase<'models', ModelItem>;
export type DataTier = TierBase<'data', DataItem>;
export type ComingTier = TierBase<'coming', ComingItem>;

/** GET /v1/capabilities (docs/API.md): six tiers, always in this order. */
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
  /** Shown above the tiers when offline; null otherwise. */
  banner: string | null;
  tiers: [AskTier, ComposedTier, BuiltinTier, ModelsTier, DataTier, ComingTier];
  guest_limits: { analyses_per_hour: number; kept_hours: number };
  session: SessionInfo;
  example_prompts: { tier: string; id: string; prompt: string }[];
}

/** GET /v1/capabilities: what this server can do. It changes only with the server's setup. */
export function useCapabilities() {
  return useQuery({
    queryKey: ['capabilities'],
    queryFn: () => apiClient.get<Capabilities>('/v1/capabilities'),
    staleTime: 5 * 60_000,
  });
}
