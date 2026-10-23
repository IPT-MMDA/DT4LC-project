import { useQuery } from '@tanstack/react-query';
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

/**
 * The session badge (design/DESIGN.md section 6), as a bold label and a line:
 * "Guest." + "4 of 5 analyses left this hour · kept 24 h · no account."
 * "Member." + "No analysis limit · assistant offline."
 */
export function sessionBadge(session: SessionInfo, assistantOffline = false): { label: string; line: string } {
  const label = `${roleLabel(session.role)}.`;
  const { limit, remaining } = session.analyses;
  if (session.guest) {
    const parts = [
      limit === null ? 'No analysis limit' : `${remaining ?? limit} of ${limit} analyses left this hour`,
      session.kept_hours !== null && `kept ${session.kept_hours} h`,
      'no account',
    ];
    return { label, line: `${parts.filter(Boolean).join(' · ')}.` };
  }
  const parts = ['No analysis limit', assistantOffline && 'assistant offline'];
  return { label, line: `${parts.filter(Boolean).join(' · ')}.` };
}
