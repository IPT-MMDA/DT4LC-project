import { describe, expect, it } from 'vitest';
import { signinNotice } from './useSigninNotice';

const notice = (query: string) => signinNotice(new URLSearchParams(query));

describe('signinNotice: the sign-in callback query on /app', () => {
  it('not on the allowlist names the email', () => {
    expect(notice('signin=denied&reason=not_allowlisted&email=olga%40outside.e2e')).toEqual({
      tone: 'denied',
      text: 'olga@outside.e2e is not on the allowlist for this server',
    });
    expect(notice('signin=denied&reason=not_allowlisted')?.text).toBe('Your GitHub email is not on the allowlist for this server');
  });

  it('an unverified primary email asks to verify it', () => {
    expect(notice('signin=denied&reason=email_unverified')).toEqual({
      tone: 'denied',
      text: 'Your primary GitHub email is not verified — verify it on GitHub, then sign in again',
    });
  });

  it('an error, or a denial for a reason this UI does not know, is "Sign-in failed — try again"', () => {
    expect(notice('signin=error')).toEqual({ tone: 'error', text: 'Sign-in failed — try again' });
    expect(notice('signin=denied&reason=other')?.text).toBe('Sign-in failed — try again');
  });

  it('nothing without signin', () => {
    expect(notice('')).toBeNull();
    expect(notice('reason=not_allowlisted&email=a%40b.c')).toBeNull();
  });
});
