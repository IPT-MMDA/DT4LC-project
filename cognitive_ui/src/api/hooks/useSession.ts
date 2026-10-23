import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../client';
import type { SessionInfo } from '../../types';

/** Invalidate this key after anything that changes the quota, e.g. an analysis starts. */
export const SESSION_QUERY_KEY = ['session'] as const;

/** GET /v1/session: role, guest quota and retention. Also issues the session cookie. */
export function useSession() {
  return useQuery({
    queryKey: SESSION_QUERY_KEY,
    queryFn: () => apiClient.get<SessionInfo>('/v1/session'),
    staleTime: 30_000,
  });
}

const ROLE_LABELS: Record<SessionInfo['role'], string> = {
  guest: 'Guest',
  member: 'Member',
  operator: 'Operator',
};

export function roleLabel(role: SessionInfo['role']): string {
  return ROLE_LABELS[role] ?? 'Guest';
}

/** What the badge says about the server: GET /v1/health and GET /v1/capabilities. */
export interface BadgeServer {
  /** `assistant.offline` from GET /v1/health; undefined until health is known (the part is left out). */
  offline?: boolean;
  /** The assistant's display name (`assistant.display` from GET /v1/capabilities), e.g. "Gemini 3.5 Flash". */
  assistant?: string | null;
  /** `summary.parts.deployment` from GET /v1/capabilities: "local server" or "public demo". */
  deployment?: string | null;
}

export interface SessionBadgeText {
  label: string;
  /** The line's parts; for a guest the first is the quota. */
  parts: string[];
  line: string;
  /** A guest with one analysis or none left: the quota part shows in the `canc` hue. */
  lowQuota: boolean;
}

const sentence = (parts: string[]) => {
  const text = parts.join(' · ');
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
};

/**
 * The session badge (design/DESIGN.md section 6), as a bold label and a line:
 * "Guest." + "4 of 5 analyses left this hour · kept 24 h · no account."
 * "Member." + "Local server · no analysis limit · assistant offline."
 * An operator reads like a member. A guest's line is the design's whether or not
 * the assistant is reachable: the composer's offline line says that.
 */
export function sessionBadge(session: SessionInfo, server: BadgeServer = {}): SessionBadgeText {
  const label = `${roleLabel(session.role)}.`;
  const { limit, remaining } = session.analyses;
  let parts: string[];
  let lowQuota = false;
  if (session.guest) {
    const left = remaining ?? limit;
    lowQuota = limit !== null && left !== null && left <= 1;
    parts = [
      limit === null ? 'no analysis limit' : `${left} of ${limit} analyses left this hour`,
      ...(session.kept_hours !== null ? [`kept ${session.kept_hours} h`] : []),
      'no account',
    ];
  } else {
    const assistant =
      server.offline === undefined
        ? []
        : [server.offline ? 'assistant offline' : `${server.assistant || 'assistant'} connected`];
    parts = [...(server.deployment ? [server.deployment] : []), 'no analysis limit', ...assistant];
  }
  return { label, parts, line: sentence(parts), lowQuota };
}

/** "14:05": a time of day in this browser's time zone. */
function localTime(iso: string): string | null {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return null;
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(at.getHours())}:${pad(at.getMinutes())}`;
}

/**
 * The composer's line for a guest with no analysis left this hour, or null:
 * "You've used your 5 analyses for this hour — the next one is available at 14:05."
 * Null once `resets_at` has passed (the session is stale; it refetches then).
 */
export function quotaLine(session: SessionInfo | undefined, now = Date.now()): string | null {
  if (!session?.guest) return null;
  const { limit, remaining, resets_at: resetsAt } = session.analyses;
  if (limit === null || remaining === null || remaining > 0) return null;
  if (resetsAt && Date.parse(resetsAt) <= now) return null;
  const used = `You've used your ${limit} analyses for this hour`;
  const at = resetsAt ? localTime(resetsAt) : null;
  return at ? `${used} — the next one is available at ${at}.` : `${used}.`;
}

/** quotaLine for this session; at `resets_at` it clears and the session refetches. */
export function useQuotaLine(): string | null {
  const { data: session } = useSession();
  const queryClient = useQueryClient();
  const [now, setNow] = useState(() => Date.now());
  const line = quotaLine(session, now);
  const resetsAt = session?.analyses.resets_at;
  useEffect(() => {
    if (!line || !resetsAt) return;
    const timer = setTimeout(() => {
      setNow(Date.now());
      void queryClient.invalidateQueries({ queryKey: SESSION_QUERY_KEY });
    }, Math.max(0, Date.parse(resetsAt) - Date.now()) + 1000);
    return () => clearTimeout(timer);
  }, [line, resetsAt, queryClient]);
  return line;
}
