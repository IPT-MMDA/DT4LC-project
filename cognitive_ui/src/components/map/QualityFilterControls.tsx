import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export function QualityFilterControls() {
  const [isExpanded, setIsExpanded] = useState(false);
  const { cloudCoverMax, setCloudCoverMax } = useAppStore();

  return (
    <div className="border border-border rounded-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-surface2 rounded-t-lg"
      >
        <span className="text-sm font-semibold text-text">Quality Filters</span>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isExpanded && (
        <div className="p-3 pt-0">
          <div className="mb-2 text-sm text-text2">
            Max Cloud Cover: {cloudCoverMax}%
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={cloudCoverMax}
            onChange={(e) => setCloudCoverMax(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
          <p className="text-xs text-muted mt-2">
            Filter out images with cloud cover above this threshold
          </p>
        </div>
      )}
    </div>
  );
}
