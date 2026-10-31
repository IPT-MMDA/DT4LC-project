import { useState } from 'react';
import type { LandCoverModel } from '../components/chat/result/resultModel';

/**
 * The active view of a land-cover card: its key, the national/local scope, and the view shown. On a hazard card
 * `landNow` shows the start map over the chosen hazard, which keeps its scope; choosing a hazard or a scope ends it.
 */
export function useLandCoverView(model: LandCoverModel) {
  const [active, setActive] = useState<string>(model.views[0]?.key ?? '');
  const [local, setLocal] = useState(false);
  const [landNow, setLandNow] = useState(false);
  const base = model.views.find((v) => v.key === active) ?? model.views[0];
  const twin = base && model.local[base.key];
  const start = landNow ? model.views.find((v) => v.key === 'start_class') : undefined;
  return {
    base,
    view: start ?? (local && twin ? twin : base),
    hasLocal: Boolean(twin),
    local,
    landNow: Boolean(start),
    setLocal: (on: boolean) => {
      setLocal(on);
      setLandNow(false);
    },
    setActive: (key: string) => {
      setActive(key);
      setLandNow(false);
    },
    setLandNow,
  };
}
