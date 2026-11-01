import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export interface SigninNotice {
  tone: 'denied' | 'error';
  text: string;
}

/** The query keys the sign-in callback adds to `/app` (server/routes/auth.py). */
const SIGNIN_PARAMS = ['signin', 'reason', 'email'];

/**
 * What the app says after GitHub sent the browser back without signing in:
 *   ?signin=denied&reason=not_allowlisted&email=a@b.c → "a@b.c is not on the allowlist for this server"
 *   ?signin=denied&reason=email_unverified            → the verified-email message
 *   ?signin=error                                     → "Sign-in failed — try again"
 * Null without `signin`.
 */
export function signinNotice(params: URLSearchParams): SigninNotice | null {
  const signin = params.get('signin');
  if (!signin) return null;
  if (signin === 'denied') {
    const reason = params.get('reason');
    if (reason === 'not_allowlisted') {
      const email = params.get('email');
      return { tone: 'denied', text: `${email || 'Your GitHub email'} is not on the allowlist for this server` };
    }
    if (reason === 'email_unverified') {
      return {
        tone: 'denied',
        text: 'Your primary GitHub email is not verified — verify it on GitHub, then sign in again',
      };
    }
  }
  return { tone: 'error', text: 'Sign-in failed — try again' };
}

/**
 * Reads the sign-in callback's query once, keeps the notice until dismissed, and replaces the
 * URL without `signin`, `reason` and `email` (other parameters stay).
 */
export function useSigninNotice(): [SigninNotice | null, () => void] {
  const { pathname, search, hash } = useLocation();
  const navigate = useNavigate();
  const [notice, setNotice] = useState<SigninNotice | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(search);
    const found = signinNotice(params);
    if (!found) return;
    setNotice(found);
    for (const key of SIGNIN_PARAMS) params.delete(key);
    const rest = params.toString();
    navigate({ pathname, search: rest ? `?${rest}` : '', hash }, { replace: true });
  }, [pathname, search, hash, navigate]);

  return [notice, () => setNotice(null)];
}
