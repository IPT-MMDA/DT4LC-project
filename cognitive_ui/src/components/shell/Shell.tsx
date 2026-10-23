import { useEffect } from 'react';
import { X } from 'lucide-react';
import { Outlet } from 'react-router-dom';
import { useSigninNotice } from '../../hooks/useSigninNotice';
import { useAppStore } from '../../store/useAppStore';
import { cn } from '../../utils/cn';
import { applyTheme } from '../../utils/theme';
import { Rail } from './Rail';

/** The app frame under /app: the rail and the current destination. */
export function Shell() {
  const theme = useAppStore((state) => state.theme);
  const [notice, dismissNotice] = useSigninNotice();

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <div className="h-screen flex overflow-hidden bg-bg text-text">
      <Rail />
      <div className="flex-1 min-w-0 h-full flex flex-col overflow-hidden">
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
