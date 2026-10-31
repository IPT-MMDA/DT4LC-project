import { useState } from 'react';
import type { LandCoverModel } from '../components/chat/result/resultModel';

/** The active view of a land-cover card: its key, the national/local scope, and the view shown. */
export function useLandCoverView(model: LandCoverModel) {
  const [active, setActive] = useState<string>(model.views[0]?.key ?? '');
  const [local, setLocal] = useState(false);
  const base = model.views.find((v) => v.key === active) ?? model.views[0];
  const twin = base && model.local[base.key];
  return { base, view: local && twin ? twin : base, hasLocal: Boolean(twin), local, setLocal, setActive };
}
