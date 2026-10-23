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
    <div className="border border-border rounded-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-surface2 rounded-t-lg"
      >
        <span className="text-sm font-semibold text-text">Time Periods (Change Detection)</span>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isExpanded && (
        <div className="p-3 pt-0 space-y-4">
          <div>
            <label className="block text-sm font-medium text-text2 mb-2">Pre-Event Period</label>
            <div className="flex items-center gap-2">
              <input
                type="date"
                value={prePeriod.start}
                onChange={(e) => setPrePeriod(e.target.value, prePeriod.end)}
                className="flex-1 px-2 py-1 text-sm border border-border rounded bg-surface"
              />
              <span className="text-sm text-muted">to</span>
              <input
                type="date"
                value={prePeriod.end}
                onChange={(e) => setPrePeriod(prePeriod.start, e.target.value)}
                className="flex-1 px-2 py-1 text-sm border border-border rounded bg-surface"
              />
            </div>
          </div>

          <div className="border-t border-border pt-4">
            <label className="block text-sm font-medium text-text2 mb-2">Post-Event Period</label>

            <label className="flex items-center gap-2 mb-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useNowForPost}
                onChange={(e) => setUseNowForPost(e.target.checked)}
              />
              <span className="text-sm text-text2">Use "Now" (last 7 days)</span>
            </label>

            {nowPeriod && useNowForPost && (
              <div className="p-2 mb-2 bg-accent-tint border border-accent/30 rounded text-xs text-accent-ink">
                Using: {nowPeriod.start} to {nowPeriod.end}
              </div>
            )}

            <div className="flex items-center gap-2">
              <input
                type="date"
                value={postPeriod.start}
                onChange={(e) => setPostPeriod(e.target.value, postPeriod.end)}
                disabled={useNowForPost}
                className="flex-1 px-2 py-1 text-sm border border-border rounded bg-surface disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <span className="text-sm text-muted">to</span>
              <input
                type="date"
                value={postPeriod.end}
                onChange={(e) => setPostPeriod(postPeriod.start, e.target.value)}
                disabled={useNowForPost}
                className="flex-1 px-2 py-1 text-sm border border-border rounded bg-surface disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
          </div>

          <p className="text-xs text-muted mt-3">
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
