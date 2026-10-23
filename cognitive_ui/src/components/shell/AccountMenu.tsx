import { useEffect, useId, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { SIGN_IN_URL, signOut } from '../../api/auth';
import { SESSION_QUERY_KEY, roleLabel, useSession } from '../../api/hooks/useSession';
import { FILES_QUERY_KEY } from '../../api/hooks/useUpload';
import { JOBS_QUERY_KEY } from '../../api/library';
import { useThemeToggle } from '../../hooks/useThemeToggle';
import type { SessionUser } from '../../types';

/** The avatar, or the first letter of the user's name (the role's for a guest). */
function Avatar({ user, fallback, size }: { user: SessionUser | null | undefined; fallback: string; size: 'button' | 'menu' }) {
  const box = size === 'button' ? 'w-[30px] h-[30px] text-[10px]' : 'w-8 h-8 text-[12px]';
  if (user?.avatar_url) return <img src={user.avatar_url} alt="" className={`${box} rounded-pill object-cover`} />;
  const letter = (user ? user.name || user.login : fallback).charAt(0).toUpperCase();
  return (
    <span aria-hidden className={`${box} rounded-pill bg-bubble text-muted font-semibold flex items-center justify-center`}>
      {letter}
    </span>
  );
}

/**
 * The account button at the bottom of the rail and its menu: who is signed in and their role
 * badge, the theme toggle, Settings for operators, and GitHub sign-in or sign-out
 * (design/DESIGN.md §1, §6). Without GITHUB_CLIENT_ID and friends the server reports
 * `sign_in.configured: false` and the menu says so instead of offering a button.
 */
export function AccountMenu() {
  const [open, setOpen] = useState(false);
  const [signOutError, setSignOutError] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuId = useId();
  const { data: session } = useSession();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const theme = useThemeToggle();
  const role = roleLabel(session?.role ?? 'guest');
  const user = session?.user ?? null;
  const who = user ? user.name || user.login : null;

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  // POST /v1/auth/logout, then everything that depends on who is asking is fetched again.
  const onSignOut = async () => {
    setSigningOut(true);
    setSignOutError(false);
    try {
      await signOut();
      await Promise.all(
        [SESSION_QUERY_KEY, ['capabilities'], JOBS_QUERY_KEY, FILES_QUERY_KEY].map((queryKey) =>
          queryClient.invalidateQueries({ queryKey }),
        ),
      );
      setOpen(false);
      navigate('/');
    } catch {
      setSignOutError(true);
    } finally {
      setSigningOut(false);
    }
  };

  const itemClass = 'w-full text-left px-3 py-[7px] rounded-control text-control text-text2 hover:bg-surface2';
  const label = who ? `Account: ${who} (${role})` : `Account: ${role}`;

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-label={label}
        title={who ?? undefined}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((v) => !v)}
        className="w-[30px] h-[30px] rounded-pill flex items-center justify-center"
      >
        <Avatar user={user} fallback={role} size="button" />
      </button>

      {open && (
        <div
          id={menuId}
          role="menu"
          aria-label="Account"
          className="absolute left-full bottom-0 ml-2 z-20 w-64 p-1 bg-surface border border-border rounded-card flex flex-col"
        >
          {user ? (
            <div data-testid="account-user" className="px-3 py-2 flex items-start gap-[10px]">
              <Avatar user={user} fallback={role} size="menu" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-control font-semibold text-text truncate">{who}</span>
                  <RoleBadge role={role} />
                </div>
                <p className="text-[11.5px] text-muted truncate">@{user.login}</p>
                <p className="text-[11.5px] text-muted truncate" title={user.email}>
                  {user.email}
                </p>
              </div>
            </div>
          ) : (
            <div className="px-3 py-2 flex items-center justify-between gap-2">
              <span className="text-control text-muted">Browsing as</span>
              {session && <RoleBadge role={role} />}
            </div>
          )}
          <div className="my-1 border-t border-border2" />
          <button type="button" role="menuitem" className={itemClass} onClick={theme.toggle}>
            {theme.isDark ? 'Light theme' : 'Dark theme'}
          </button>
          {session?.role === 'operator' && (
            <Link to="/app/settings" role="menuitem" className={itemClass} onClick={() => setOpen(false)}>
              Settings
            </Link>
          )}
          {session && !user && session.sign_in?.configured && (
            // A full navigation: the server redirects to GitHub, and GitHub back to /app.
            <a href={SIGN_IN_URL} role="menuitem" className={`${itemClass} font-medium text-accent-ink`}>
              Sign in with GitHub
            </a>
          )}
          {session && !user && !session.sign_in?.configured && (
            <p data-testid="sign-in-unavailable" className="px-3 py-[7px] text-control text-faint">
              Sign-in not configured on this server
            </p>
          )}
          {user && (
            <button type="button" role="menuitem" className={itemClass} disabled={signingOut} onClick={() => void onSignOut()}>
              {signingOut ? 'Signing out…' : 'Sign out'}
            </button>
          )}
          {signOutError && (
            <p role="alert" className="px-3 py-[5px] text-[11.5px] text-fail-ink">
              Could not sign out — try again.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function RoleBadge({ role }: { role: string }) {
  return (
    <span data-testid="role-badge" className="flex-none px-2 py-[2px] rounded-pill bg-accent-tint text-accent-ink text-[11px] font-semibold">
      {role}
    </span>
  );
}
