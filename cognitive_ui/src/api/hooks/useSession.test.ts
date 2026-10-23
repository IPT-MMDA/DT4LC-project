import { describe, expect, it } from 'vitest';
import { GUEST_SESSION, MEMBER_SESSION, OPERATOR_SESSION } from '../../test/fixtures';
import { roleLabel, sessionBadge } from './useSession';

describe('sessionBadge (DESIGN §6)', () => {
  it('guest: analyses left, retention, no account', () => {
    expect(sessionBadge(GUEST_SESSION)).toEqual({
      label: 'Guest.',
      line: '4 of 5 analyses left this hour · kept 24 h · no account.',
    });
    const fresh = { ...GUEST_SESSION, analyses: { limit: 5, used: 0, remaining: 5, resets_at: null } };
    expect(sessionBadge(fresh).line).toBe('5 of 5 analyses left this hour · kept 24 h · no account.');
  });

  it('member: no limit, and the assistant state when it is offline', () => {
    expect(sessionBadge(MEMBER_SESSION, true)).toEqual({ label: 'Member.', line: 'No analysis limit · assistant offline.' });
    expect(sessionBadge(MEMBER_SESSION, false).line).toBe('No analysis limit.');
  });

  it('operator', () => {
    expect(sessionBadge(OPERATOR_SESSION).label).toBe('Operator.');
    expect(roleLabel('operator')).toBe('Operator');
  });
});
