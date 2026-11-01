import apiClient, { API_BASE_URL } from './client';
import type { SessionInfo } from '../types';

/**
 * GitHub sign-in starts with a full navigation here (not a fetch): the server answers 302 to
 * GitHub, and GitHub's callback 302s back to `/app`, with `?signin=denied|error` when it refused.
 */
export const SIGN_IN_URL = `${API_BASE_URL}/v1/auth/github/login`;

/** POST /v1/auth/logout: the session cookie becomes a fresh guest session, which it returns. */
export function signOut(): Promise<SessionInfo> {
  return apiClient.post<undefined, SessionInfo>('/v1/auth/logout');
}
