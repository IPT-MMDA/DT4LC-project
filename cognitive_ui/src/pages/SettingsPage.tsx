import { useAppStore } from '../store/useAppStore';

export function SettingsPage() {
  const { theme, setTheme } = useAppStore();

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-text">
          Settings
        </h1>
        <p className="text-muted mt-1">
          Manage your preferences and configuration
        </p>
      </div>

      <div className="bg-surface rounded-lg border border-border p-6">
        <h2 className="text-xl font-semibold text-text mb-4">
          Appearance
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text2 mb-2">
              Theme
            </label>
            <select
              value={theme}
              onChange={(e) =>
                setTheme(e.target.value as 'light' | 'dark' | 'auto')
              }
              className="px-4 py-2 bg-surface2 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-accent text-text"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="auto">Auto</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
