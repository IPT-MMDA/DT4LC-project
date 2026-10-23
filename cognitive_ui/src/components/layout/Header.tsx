import { Menu, Moon, Sun, Settings } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { useHealth } from '../../api/hooks/useHealth';

export function Header() {
  const { theme, setTheme, toggleSidebar } = useAppStore();
  const { data: health } = useHealth();

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <header className="h-16 border-b border-border bg-surface flex items-center justify-between px-4">
      <div className="flex items-center gap-4">
        <button
          onClick={toggleSidebar}
          className="p-2 hover:bg-surface2 rounded-lg"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-semibold">DT4LC</h1>
          {health?.ok && (
            <span className="w-2 h-2 bg-done rounded-full"></span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={toggleTheme}
          className="p-2 hover:bg-surface2 rounded-lg"
        >
          {theme === 'light' ? (
            <Moon className="w-5 h-5" />
          ) : (
            <Sun className="w-5 h-5" />
          )}
        </button>
        <button className="p-2 hover:bg-surface2 rounded-lg">
          <Settings className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
