import { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export function TimePeriodSelector() {
  const [isExpanded, setIsExpanded] = useState(true);
  const { prePeriod, postPeriod, useNowForPost, setPrePeriod, setPostPeriod, setUseNowForPost } = useAppStore();

  useEffect(() => {
    if (useNowForPost) {
      const nowPeriod = calculateNowPeriod();
      setPostPeriod(nowPeriod.start, nowPeriod.end);
    }
  }, [useNowForPost, setPostPeriod]);

  const nowPeriod = useNowForPost ? calculateNowPeriod() : null;

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-t-lg"
      >
        <span className="text-sm font-semibold text-gray-900 dark:text-white">Time Periods (Change Detection)</span>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isExpanded && (
        <div className="p-3 pt-0 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Pre-Event Period</label>
            <div className="flex items-center gap-2">
              <input
                type="date"
                value={prePeriod.start}
                onChange={(e) => setPrePeriod(e.target.value, prePeriod.end)}
                className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
              />
              <span className="text-sm text-gray-500">to</span>
              <input
                type="date"
                value={prePeriod.end}
                onChange={(e) => setPrePeriod(prePeriod.start, e.target.value)}
                className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
              />
            </div>
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Post-Event Period</label>

            <label className="flex items-center gap-2 mb-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useNowForPost}
                onChange={(e) => setUseNowForPost(e.target.checked)}
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Use "Now" (last 7 days)</span>
            </label>

            {nowPeriod && useNowForPost && (
              <div className="p-2 mb-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded text-xs text-blue-800 dark:text-blue-200">
                Using: {nowPeriod.start} to {nowPeriod.end}
              </div>
            )}

            <div className="flex items-center gap-2">
              <input
                type="date"
                value={postPeriod.start}
                onChange={(e) => setPostPeriod(e.target.value, postPeriod.end)}
                disabled={useNowForPost}
                className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <span className="text-sm text-gray-500">to</span>
              <input
                type="date"
                value={postPeriod.end}
                onChange={(e) => setPostPeriod(postPeriod.start, e.target.value)}
                disabled={useNowForPost}
                className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
          </div>

          <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">
            Change detection will compare data between these two time periods
          </p>
        </div>
      )}
    </div>
  );
}

function calculateNowPeriod(): { start: string; end: string } {
  const today = new Date();
  const sevenDaysAgo = new Date(today);
  sevenDaysAgo.setDate(today.getDate() - 7);
  return {
    start: sevenDaysAgo.toISOString().split('T')[0],
    end: today.toISOString().split('T')[0],
  };
}
