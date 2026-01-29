import { Square, Trash2, Hand } from 'lucide-react';

interface DrawToolbarProps {
  onModeChange: (mode: 'select' | 'draw' | 'delete') => void;
  currentMode: 'select' | 'draw' | 'delete';
}

export function DrawToolbar({ onModeChange, currentMode }: DrawToolbarProps) {
  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20">
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-2 flex gap-2">
        <button
          onClick={() => onModeChange('select')}
          className={`p-3 rounded-lg transition-colors ${
            currentMode === 'select'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
          }`}
          title="Select mode - Pan and zoom the map"
          aria-label="Select mode"
        >
          <Hand className="w-5 h-5" />
        </button>

        <button
          onClick={() => onModeChange('draw')}
          className={`p-3 rounded-lg transition-colors ${
            currentMode === 'draw'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
          }`}
          title="Draw polygon - Click points to draw, double-click to finish"
          aria-label="Draw polygon"
        >
          <Square className="w-5 h-5" />
        </button>

        <button
          onClick={() => onModeChange('delete')}
          className={`p-3 rounded-lg transition-colors ${
            currentMode === 'delete'
              ? 'bg-red-500 text-white'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
          }`}
          title="Delete selected - Click a region first, then click this to delete"
          aria-label="Delete selected region"
        >
          <Trash2 className="w-5 h-5" />
        </button>
      </div>

      {currentMode === 'draw' && (
        <div className="mt-2 bg-blue-50 dark:bg-blue-950 rounded-lg shadow-lg p-3 text-sm text-blue-900 dark:text-blue-100 text-center max-w-md">
          <p className="font-medium">Drawing Mode Active</p>
          <p className="text-xs mt-1">
            Click to place points, <strong>double-click</strong> to finish polygon, or press <kbd className="px-1 py-0.5 bg-white dark:bg-gray-800 rounded border border-blue-200 dark:border-blue-700">Esc</kbd> to cancel
          </p>
        </div>
      )}
    </div>
  );
}
