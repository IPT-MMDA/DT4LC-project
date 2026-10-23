import { Link, useLocation } from 'react-router-dom';
import { Contrast, LayoutGrid, Library, Map as MapIcon, MessageSquare, type LucideIcon } from 'lucide-react';
import { useThemeToggle } from '../../hooks/useThemeToggle';
import { cn } from '../../utils/cn';
import { AccountMenu } from './AccountMenu';

interface RailItem {
  to: string;
  label: string;
  icon: LucideIcon;
  isActive: (pathname: string) => boolean;
}

// The four destinations of design/DESIGN.md section 1. A result opens in the Workspace.
const ITEMS: RailItem[] = [
  { to: '/app', label: 'Chat', icon: MessageSquare, isActive: (p) => p === '/app' || p.startsWith('/app/results') },
  { to: '/app/map', label: 'Map', icon: MapIcon, isActive: (p) => p.startsWith('/app/map') },
  { to: '/app/library', label: 'Library', icon: Library, isActive: (p) => p.startsWith('/app/library') },
  { to: '/app/capabilities', label: 'Can do', icon: LayoutGrid, isActive: (p) => p.startsWith('/app/capabilities') },
];

export function Rail() {
  const { pathname } = useLocation();
  const theme = useThemeToggle();

  return (
    <nav
      aria-label="Main"
      className="w-[60px] flex-none bg-surface border-r border-border flex flex-col items-center py-[14px] gap-[6px]"
    >
      <Link
        to="/"
        aria-label="DT4LC home"
        className="w-[30px] h-[30px] mb-[14px] rounded-[7px] bg-ink text-ink-bg font-bold text-[11px] flex items-center justify-center"
      >
        DT
      </Link>

      {ITEMS.map(({ to, label, icon: Icon, isActive }) => {
        const active = isActive(pathname);
        return (
          <Link
            key={to}
            to={to}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'w-11 h-11 rounded-[8px] flex flex-col items-center justify-center gap-[3px]',
              active ? 'bg-accent-tint text-accent-ink font-semibold' : 'text-muted hover:bg-surface2',
            )}
          >
            <Icon aria-hidden className="w-4 h-4" strokeWidth={1.8} />
            <span className="text-[9px] leading-none">{label}</span>
          </Link>
        );
      })}

      <div className="flex-1" />

      <button
        type="button"
        onClick={theme.toggle}
        aria-label={theme.label}
        title={theme.label}
        className="w-8 h-8 mb-1 rounded-control text-muted hover:bg-surface2 flex items-center justify-center"
      >
        <Contrast aria-hidden className="w-4 h-4" strokeWidth={1.8} />
      </button>

      <AccountMenu />
    </nav>
  );
}
