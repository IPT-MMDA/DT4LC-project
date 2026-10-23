// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { landingDecision, type LandingDecision, type LandingInputs } from './landingDecision';

const base: LandingInputs = { role: undefined, sessionPending: false, hasConversations: false, deepLink: false };

describe('landingDecision: who sees the landing at /', () => {
  it.each<[LandingInputs['role'], boolean, LandingDecision]>([
    ['guest', false, 'landing'],
    ['guest', true, 'app'],
    ['member', false, 'app'],
    ['member', true, 'app'],
    ['operator', false, 'app'],
    ['operator', true, 'app'],
  ])('%s, conversations %s: %s', (role, hasConversations, expected) => {
    expect(landingDecision({ ...base, role, hasConversations })).toBe(expected);
  });

  it('waits for the session when nothing else decides it', () => {
    expect(landingDecision({ ...base, sessionPending: true })).toBe('wait');
  });

  it('does not wait for the session when this browser has conversations', () => {
    expect(landingDecision({ ...base, sessionPending: true, hasConversations: true })).toBe('app');
  });

  it('a failed session request counts as a guest', () => {
    expect(landingDecision({ ...base, role: undefined })).toBe('landing');
  });

  it('a Workspace deep link (?sample=, ?prompt=) always opens the Workspace', () => {
    expect(landingDecision({ ...base, role: 'guest', deepLink: true })).toBe('app');
    expect(landingDecision({ ...base, sessionPending: true, deepLink: true })).toBe('app');
  });
});
