// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { landingDecision } from './landingDecision';

// The role, the stored conversations and a pending session are no longer inputs: a
// guest, a returning guest, a member and an operator all get the landing at `/`
// before the session answers (pages/Landing.test.tsx, "/ routing").
describe('landingDecision: what / shows', () => {
  it.each(['', '?', '?utm=x', '?from=e2e&ref=rail'])('%j: the landing', (search) => {
    expect(landingDecision(search)).toBe('landing');
  });

  it.each(['?sample=1', '?sample=0', '?prompt=Map+water+change', '?prompt=', '?utm=x&sample=1'])(
    'a Workspace deep link %j: the Workspace',
    (search) => {
      expect(landingDecision(search)).toBe('app');
    },
  );
});
