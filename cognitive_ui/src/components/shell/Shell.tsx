import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';
import { applyTheme } from '../../utils/theme';
import { Rail } from './Rail';

/** The app frame under /app: the rail and the current destination. */
export function Shell() {
  const theme = useAppStore((state) => state.theme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <div className="h-screen flex overflow-hidden bg-bg text-text">
      <Rail />
      <div className="flex-1 min-w-0 h-full flex overflow-hidden">
        <Outlet />
      </div>
    </div>
  );
}
