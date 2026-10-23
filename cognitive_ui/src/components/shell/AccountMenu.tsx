import { useEffect, useId, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { roleLabel, useSession } from '../../api/hooks/useSession';
import { useThemeToggle } from '../../hooks/useThemeToggle';

/**
 * The account button at the bottom of the rail and its menu: role badge, theme
 * toggle, sign in (Phase 5) and, for operators only, Settings (design/DESIGN.md §1, §6).
 */
export function AccountMenu() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuId = useId();
  const { data: session } = useSession();
  const theme = useThemeToggle();
  const role = roleLabel(session?.role ?? 'guest');

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

  const itemClass = 'w-full text-left px-3 py-[7px] rounded-control text-control text-text2 hover:bg-surface2';

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-label={`Account: ${role}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((v) => !v)}
        className="w-[30px] h-[30px] rounded-pill bg-bubble text-muted text-[10px] font-semibold flex items-center justify-center"
      >
        {role[0]}
      </button>

      {open && (
        <div
          id={menuId}
          role="menu"
          aria-label="Account"
          className="absolute left-full bottom-0 ml-2 z-20 w-56 p-1 bg-surface border border-border rounded-card flex flex-col"
        >
          <div className="px-3 py-2 flex items-center justify-between gap-2">
            <span className="text-control text-muted">Browsing as</span>
            {session && (
              <span
                data-testid="role-badge"
                className="px-2 py-[2px] rounded-pill bg-accent-tint text-accent-ink text-[11px] font-semibold"
              >
                {role}
              </span>
            )}
          </div>
          <div className="my-1 border-t border-border2" />
          <button type="button" role="menuitem" className={itemClass} onClick={theme.toggle}>
            {theme.isDark ? 'Light theme' : 'Dark theme'}
          </button>
          {session?.role === 'operator' && (
            <Link to="/app/settings" role="menuitem" className={itemClass} onClick={() => setOpen(false)}>
              Settings
            </Link>
          )}
          <button
            type="button"
            role="menuitem"
            disabled
            aria-describedby={`${menuId}-soon`}
            className="w-full text-left px-3 py-[7px] rounded-control text-control text-faint flex justify-between cursor-not-allowed"
          >
            <span>Sign in</span>
            <span id={`${menuId}-soon`} className="text-[11px]">
              Coming soon
            </span>
          </button>
        </div>
      )}
    </div>
  );
}
