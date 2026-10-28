import { useEffect, useRef, type KeyboardEvent, type Ref } from 'react';
import { Contrast, Menu, X } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { useThemeToggle } from '../../hooks/useThemeToggle';
import { useChatStore } from '../../store/useChatStore';
import { cn } from '../../utils/cn';
import { AccountMenu } from './AccountMenu';
import { ConversationList } from './ConversationList';
import { NAV_ITEMS, screenTitle } from './navigation';

const MENU_ID = 'phone-menu';

/**
 * The phone's top bar, in place of the rail:
 * the menu button, the screen's title (the conversation's in the Workspace) and, in the
 * Workspace, Map ↗.
 */
export function PhoneBar({ open, onOpen, buttonRef }: { open: boolean; onOpen: () => void; buttonRef: Ref<HTMLButtonElement> }) {
  const { pathname } = useLocation();
  const conversationTitle = useChatStore((state) => state.conversations.find((c) => c.id === state.activeConversationId)?.title);
  const workspace = pathname === '/app';
  // The Workspace's own header is hidden on a phone, so this title is its heading.
  const Title = workspace ? 'h1' : 'span';

  return (
    <header data-testid="phone-bar" className="h-[52px] flex-none flex items-center gap-2 pl-2 pr-4 bg-surface border-b border-border">
      <button
        ref={buttonRef}
        type="button"
        onClick={onOpen}
        aria-label="Open menu"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? MENU_ID : undefined}
        className="w-10 h-10 flex-none flex items-center justify-center rounded-control text-text hover:bg-surface2"
      >
        <Menu aria-hidden className="w-5 h-5" strokeWidth={1.8} />
      </button>
      <Title className="flex-1 min-w-0 truncate text-card">{screenTitle(pathname, conversationTitle)}</Title>
      {workspace && (
        <Link to="/app/map" className="flex-none text-control font-medium text-accent-ink">
          Map ↗
        </Link>
      )}
    </header>
  );
}

/**
 * The phone's menu: a drawer from the left with the four destinations, the conversations
 * (in the Workspace), the theme and the account menu. Escape, Close, a destination or a
 * conversation (the current one too) and a tap beside it close it.
 */
export function MenuDrawer({ onClose }: { onClose: () => void }) {
  const { pathname } = useLocation();
  const theme = useThemeToggle();
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const workspace = pathname === '/app';

  // Tab stays in the drawer; focus goes back to the menu button however it closes.
  useFocusTrap(dialogRef);
  useEffect(() => {
    closeRef.current?.focus();
  }, []);

  // Escape closes the innermost: an open popup inside (the account menu) first.
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Escape') return;
    if (event.currentTarget.querySelector('[aria-expanded="true"]')) return;
    onClose();
  };

  return (
    <div className="fixed inset-0 z-40 flex">
      <div
        ref={dialogRef}
        id={MENU_ID}
        role="dialog"
        aria-modal="true"
        aria-label="Menu"
        onKeyDown={onKeyDown}
        className="w-[min(320px,86vw)] h-full flex-none bg-surface border-r border-border flex flex-col"
      >
        <div className="h-[52px] flex-none flex items-center gap-[10px] pl-4 pr-2 border-b border-border">
          <Link
            to="/"
            aria-label="DT, DT4LC home"
            className="w-[30px] h-[30px] rounded-[7px] bg-ink text-ink-bg font-bold text-[11px] flex items-center justify-center"
          >
            DT
          </Link>
          <span className="flex-1 font-semibold">DT4LC</span>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close menu"
            className="w-10 h-10 flex items-center justify-center rounded-control text-muted hover:bg-surface2"
          >
            <X aria-hidden className="w-5 h-5" strokeWidth={1.8} />
          </button>
        </div>

        <nav aria-label="Main" className="flex-none p-2 flex flex-col gap-[2px] border-b border-border">
          {NAV_ITEMS.map(({ to, label, icon: Icon, isActive }) => {
            const active = isActive(pathname);
            return (
              <Link
                key={to}
                to={to}
                aria-current={active ? 'page' : undefined}
                onClick={onClose}
                className={cn(
                  'h-10 px-3 rounded-[8px] flex items-center gap-3 text-[13.5px]',
                  active ? 'bg-accent-tint text-accent-ink font-semibold' : 'text-text2 hover:bg-surface2',
                )}
              >
                <Icon aria-hidden className="w-4 h-4" strokeWidth={1.8} />
                {label}
              </Link>
            );
          })}
        </nav>

        {workspace ? <ConversationList className="w-auto flex-1 border-r-0" onPick={onClose} /> : <div className="flex-1" />}

        <div className="flex-none border-t border-border px-3 py-2 flex items-center gap-2">
          <AccountMenu placement="up" />
          <div className="flex-1" />
          <button
            type="button"
            onClick={theme.toggle}
            className="h-9 px-3 rounded-control text-control text-text2 hover:bg-surface2 flex items-center gap-2"
          >
            <Contrast aria-hidden className="w-4 h-4" strokeWidth={1.8} />
            {theme.isDark ? 'Light theme' : 'Dark theme'}
          </button>
        </div>
      </div>
      {/* The screen beside the drawer: a tap closes it (the Close button is the control). */}
      <div aria-hidden className="flex-1 bg-bg/70" onClick={onClose} />
    </div>
  );
}
