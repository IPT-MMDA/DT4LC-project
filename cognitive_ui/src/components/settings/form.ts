import type { SettingsDoc } from '../../api/settings';
import { fieldOf, valueOf, type Edits, type Field, type FieldValue } from './settingsModel';

/** The page's draft and its actions, handed to every section. */
export interface SettingsForm {
  doc: SettingsDoc;
  edits: Edits;
  errors: Record<string, string[]>;
  field: (path: string) => Field;
  value: <T extends FieldValue>(path: string) => T;
  set: (path: string, value: FieldValue) => void;
  reset: (path: string) => void;
  undo: (path: string) => void;
}

export function makeForm(
  doc: SettingsDoc,
  edits: Edits,
  errors: Record<string, string[]>,
  actions: Pick<SettingsForm, 'set' | 'reset' | 'undo'>,
): SettingsForm {
  return {
    doc,
    edits,
    errors,
    field: (path) => fieldOf(doc, path),
    value: (path) => valueOf(doc, edits, path),
    ...actions,
  };
}

export const card = 'border border-border rounded-card bg-surface px-4 py-[14px] flex flex-col';
export const inputClass =
  'px-[10px] py-[6px] rounded-control border border-border bg-surface text-control text-text placeholder:text-faint focus:outline-none focus:border-accent disabled:text-faint';
export const smallButton =
  'px-[9px] py-[4px] border border-border rounded-control text-[12px] text-text2 whitespace-nowrap enabled:hover:bg-surface2 disabled:text-faint disabled:cursor-not-allowed';

