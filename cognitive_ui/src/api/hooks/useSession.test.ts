import { afterEach, describe, expect, it, vi } from 'vitest';
import { GUEST_SESSION, MEMBER_SESSION, OPERATOR_SESSION } from '../../test/fixtures';
import type { SessionInfo } from '../../types';
import { quotaLine, roleLabel, sessionBadge, type BadgeServer } from './useSession';

const LOCAL_ONLINE: BadgeServer = { offline: false, assistant: 'Gemini 3.5 Flash', deployment: 'local server' };
const LOCAL_OFFLINE: BadgeServer = { offline: true, assistant: null, deployment: 'local server' };
const guestWith = (remaining: number, resetsAt: string | null = GUEST_SESSION.analyses.resets_at): SessionInfo => ({
  ...GUEST_SESSION,
  analyses: { limit: 5, used: 5 - remaining, remaining, resets_at: resetsAt },
});

afterEach(() => {
  vi.useRealTimers();
});

describe('sessionBadge', () => {
  it('guest: analyses left, retention, no account; the same online and offline', () => {
    for (const server of [LOCAL_ONLINE, LOCAL_OFFLINE, {}]) {
      expect(sessionBadge(GUEST_SESSION, server)).toMatchObject({
        label: 'Guest.',
        line: '4 of 5 analyses left this hour · kept 24 h · no account.',
        lowQuota: false,
      });
    }
    expect(sessionBadge(guestWith(5)).line).toBe('5 of 5 analyses left this hour · kept 24 h · no account.');
    expect(sessionBadge({ ...GUEST_SESSION, kept_hours: 6 }).line).toBe('4 of 5 analyses left this hour · kept 6 h · no account.');
  });

  it('member: the server, no limit and the assistant from health', () => {
    expect(sessionBadge(MEMBER_SESSION, LOCAL_OFFLINE)).toMatchObject({
      label: 'Member.',
      line: 'Local server · no analysis limit · assistant offline.',
      lowQuota: false,
    });
    expect(sessionBadge(MEMBER_SESSION, LOCAL_ONLINE).line).toBe('Local server · no analysis limit · Gemini 3.5 Flash connected.');
    expect(sessionBadge(MEMBER_SESSION, { ...LOCAL_ONLINE, deployment: 'public demo' }).line).toBe(
      'Public demo · no analysis limit · Gemini 3.5 Flash connected.',
    );
  });

  it('member: what is not known yet is left out', () => {
    expect(sessionBadge(MEMBER_SESSION, { offline: true }).line).toBe('No analysis limit · assistant offline.');
    expect(sessionBadge(MEMBER_SESSION, { offline: false }).line).toBe('No analysis limit · assistant connected.');
    expect(sessionBadge(MEMBER_SESSION).line).toBe('No analysis limit.');
  });

  it('operator: as a member', () => {
    expect(sessionBadge(OPERATOR_SESSION, LOCAL_OFFLINE)).toMatchObject({
      label: 'Operator.',
      line: 'Local server · no analysis limit · assistant offline.',
    });
    expect(sessionBadge(OPERATOR_SESSION, LOCAL_ONLINE).line).toBe('Local server · no analysis limit · Gemini 3.5 Flash connected.');
    expect(roleLabel('operator')).toBe('Operator');
  });

  it('a guest with one analysis or none left: the quota part is flagged for the canc hue', () => {
    expect(sessionBadge(guestWith(2)).lowQuota).toBe(false);
    expect(sessionBadge(guestWith(1))).toMatchObject({
      parts: ['1 of 5 analyses left this hour', 'kept 24 h', 'no account'],
      lowQuota: true,
    });
    expect(sessionBadge(guestWith(0))).toMatchObject({ line: '0 of 5 analyses left this hour · kept 24 h · no account.', lowQuota: true });
  });
});

describe('quotaLine: the composer at 0 analyses left', () => {
  // 14:05 in this machine's time zone, whatever it is.
  const resetsAt = new Date(2026, 9, 23, 14, 5).toISOString();
  const now = new Date(2026, 9, 23, 13, 40).getTime();

  it('none at 1 left, or for members', () => {
    expect(quotaLine(guestWith(1, resetsAt), now)).toBeNull();
    expect(quotaLine(MEMBER_SESSION, now)).toBeNull();
    expect(quotaLine(undefined, now)).toBeNull();
  });

  it('at 0: the local time of resets_at', () => {
    expect(quotaLine(guestWith(0, resetsAt), now)).toBe(
      "You've used your 5 analyses for this hour — the next one is available at 14:05.",
    );
    expect(quotaLine(guestWith(0, null), now)).toBe("You've used your 5 analyses for this hour.");
  });

  it('gone once resets_at has passed (on a fixed clock)', () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date(2026, 9, 23, 14, 4));
    expect(quotaLine(guestWith(0, resetsAt))).not.toBeNull();
    vi.setSystemTime(new Date(2026, 9, 23, 14, 5));
    expect(quotaLine(guestWith(0, resetsAt))).toBeNull();
  });
});
