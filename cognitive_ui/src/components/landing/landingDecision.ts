import type { SessionInfo } from '../../types';

/** What `/` shows: nothing yet, the guest landing, or a redirect to the Workspace. */
export type LandingDecision = 'wait' | 'landing' | 'app';

export interface LandingInputs {
  /** The role from GET /v1/session; undefined while it loads or when it failed. */
  role: SessionInfo['role'] | undefined;
  /** GET /v1/session has not answered yet. */
  sessionPending: boolean;
  /** This browser holds conversations (`dt4lc-chat-storage`), or an active one. */
  hasConversations: boolean;
  /** `/` was opened with a Workspace deep link (`?sample=` or `?prompt=`). */
  deepLink: boolean;
}

/**
 * The landing is for first-time guests (RUN2 "Landing routing rule"): role guest and
 * no conversations in this browser. Returning guests, members, operators and deep
 * links go to `/app`. A known answer is given without waiting for the session; while
 * the session loads nothing renders, so neither page flashes. A failed session
 * request counts as a guest: the landing needs no server.
 */
export function landingDecision({ role, sessionPending, hasConversations, deepLink }: LandingInputs): LandingDecision {
  if (hasConversations || deepLink) return 'app';
  if (sessionPending) return 'wait';
  return role === 'member' || role === 'operator' ? 'app' : 'landing';
}
