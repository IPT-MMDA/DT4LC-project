import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { Outlet, useLocation } from 'react-router-dom';
import { useLayout } from '../../hooks/useMediaQuery';
import { useSigninNotice } from '../../hooks/useSigninNotice';
import { useChatStore } from '../../store/useChatStore';
import { cn } from '../../utils/cn';
import { MenuDrawer, PhoneBar } from './PhoneMenu';
import { Rail } from './Rail';

/**
 * The app frame under /app: the rail and the current destination. On a phone the rail
 * becomes a top bar whose menu button opens a drawer (design/screens/workspace-phone-390.png).
 * The frame is the dynamic viewport's height, so the composer stays above a phone's
 * keyboard and browser bars.
 */
export function Shell() {
  const [notice, dismissNotice] = useSigninNotice();
  const phone = useLayout() === 'phone';
  const [menuOpen, setMenuOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const { pathname } = useLocation();
  const activeConversation = useChatStore((state) => state.activeConversationId);

  // A new destination or conversation closes the menu.
  useEffect(() => {
    setMenuOpen(false);
  }, [pathname, activeConversation]);

  const closeMenu = () => {
    setMenuOpen(false);
    menuButton.current?.focus();
  };

  return (
    <div className={cn('h-dvh flex overflow-hidden bg-bg text-text', phone && 'flex-col')}>
      {phone ? <PhoneBar open={menuOpen} onOpen={() => setMenuOpen(true)} buttonRef={menuButton} /> : <Rail />}
      {phone && menuOpen && <MenuDrawer onClose={closeMenu} />}
      <div className="flex-1 min-w-0 min-h-0 h-full flex flex-col overflow-hidden">
        {notice && (
          <div
            role="status"
            data-testid="signin-notice"
            data-tone={notice.tone}
            className={cn(
              'flex-none pl-4 pr-2 py-1 border-b border-border text-control leading-[1.5] flex items-center gap-2',
              notice.tone === 'error' ? 'bg-fail-tint text-fail-ink' : 'bg-canc-tint text-canc-ink',
            )}
          >
            <span className="flex-1">{notice.text}</span>
            <button
              type="button"
              aria-label="Dismiss"
              onClick={dismissNotice}
              className="flex-none w-7 h-7 rounded-control flex items-center justify-center hover:bg-surface2"
            >
              <X aria-hidden className="w-[14px] h-[14px]" />
            </button>
          </div>
        )}
        <div className="flex-1 min-h-0 flex overflow-hidden">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
