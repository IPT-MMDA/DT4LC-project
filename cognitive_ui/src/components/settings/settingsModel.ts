import type { SettingsDoc, SettingSource, SettingsUpdate, SettingsStorage } from '../../api/settings';

// The Settings page's draft: a map from a field path to its new value, or `null` for
// "Reset to .env" (PUT sends null, which drops the override). Paths are the PUT body's:
//   access.guest_analyses_per_hour, models.download_policy, assistant.provider_order,
//   assistant.providers.<name>.enabled, assistant.providers.<name>.models
// A field the operator changed back to the saved value leaves the draft, so the PUT
// carries only the fields that differ.
export type FieldValue = string[] | boolean | number | string;
export type Edits = Record<string, FieldValue | null>;

export const SECTION_IDS = ['assistant', 'models', 'access', 'data_sources', 'server'] as const;
export type SectionId = (typeof SECTION_IDS)[number];

export const SECTION_LABELS: Record<SectionId, string> = {
  assistant: 'Assistant',
  models: 'Models',
  access: 'Access',
  data_sources: 'Data sources',
  server: 'Server',
};

export interface Field {
  value: FieldValue | undefined;
  editable: boolean;
  source: SettingSource | undefined;
}

type Holder = { editable: Record<string, boolean>; source?: Record<string, SettingSource> } & Record<string, unknown>;

/** The object holding `path`'s value, and the field's name in it. */
function locate(doc: SettingsDoc, path: string): { holder: Holder | undefined; field: string } {
  const parts = path.split('.');
  if (parts[0] === 'assistant' && parts[1] === 'providers' && parts.length === 4) {
    return { holder: doc.assistant.providers[parts[2]] as unknown as Holder | undefined, field: parts[3] };
  }
  return { holder: (doc as unknown as Record<string, Holder | undefined>)[parts[0]], field: parts.slice(1).join('.') };
}

/** The saved value of `path`, whether PUT takes it, and where it comes from. */
export function fieldOf(doc: SettingsDoc, path: string): Field {
  const { holder, field } = locate(doc, path);
  return {
    value: holder?.[field] as FieldValue | undefined,
    editable: holder?.editable[field] === true,
    source: holder?.source?.[field],
  };
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** What the page shows for `path`: the draft value, else the saved one (also while a reset is pending). */
export function valueOf<T extends FieldValue>(doc: SettingsDoc, edits: Edits, path: string): T {
  const edit = edits[path];
  return (edit !== undefined && edit !== null ? edit : fieldOf(doc, path).value) as T;
}

/** `edits` with `path` set to `value`; back to the saved value means no edit. */
export function setEdit(doc: SettingsDoc, edits: Edits, path: string, value: FieldValue): Edits {
  const next = { ...edits };
  if (same(fieldOf(doc, path).value, value)) delete next[path];
  else next[path] = value;
  return next;
}

/** `edits` with `path` marked "Reset to .env": the PUT sends null for it. */
export function resetEdit(edits: Edits, path: string): Edits {
  return { ...edits, [path]: null };
}

/** `edits` without `path`: undoes a change or a pending reset. */
export function undoEdit(edits: Edits, path: string): Edits {
  const next = { ...edits };
  delete next[path];
  return next;
}

/** The PUT body: only the edited fields, nested as SettingsUpdate. */
export function buildUpdate(edits: Edits): SettingsUpdate {
  const update: Record<string, Record<string, unknown>> = {};
  for (const [path, value] of Object.entries(edits)) {
    const parts = path.split('.');
    const section = (update[parts[0]] ??= {});
    if (parts[0] === 'assistant' && parts[1] === 'providers' && parts.length === 4) {
      const providers = (section.providers ??= {}) as Record<string, Record<string, unknown>>;
      (providers[parts[2]] ??= {})[parts[3]] = value;
    } else {
      section[parts.slice(1).join('.')] = value;
    }
  }
  return update as SettingsUpdate;
}

/** The sections that have unsaved changes, for the dots in the section list. */
export function dirtySections(edits: Edits): Set<string> {
  return new Set(Object.keys(edits).map((path) => path.split('.')[0]));
}

interface ValidationDetail {
  loc?: (string | number)[];
  msg?: string;
}

/**
 * A 422's FastAPI `detail[]` by field path: `["body", "access", "allowed_domains", 1]` is
 * `access.allowed_domains` with "Item 2: …". A section-level error (`["body", "assistant"]`)
 * is under the section id; anything without a usable `loc` is under "".
 */
export function errorsByField(body: unknown): Record<string, string[]> {
  const detail = (body as { detail?: unknown } | null | undefined)?.detail;
  const out: Record<string, string[]> = {};
  const add = (path: string, msg: string) => (out[path] ??= []).push(msg);
  if (!Array.isArray(detail)) {
    if (typeof detail === 'string') add('', detail);
    return out;
  }
  for (const item of detail as ValidationDetail[]) {
    const loc = (item.loc ?? []).filter((part, i) => !(i === 0 && part === 'body'));
    let index: number | undefined;
    while (loc.length && typeof loc[loc.length - 1] === 'number') index = loc.pop() as number;
    const msg = (item.msg ?? 'Invalid value').replace(/^Value error, /, '');
    add(loc.join('.'), index === undefined ? msg : `Item ${index + 1}: ${msg}`);
  }
  return out;
}

/** Every path the page renders an input for (and so shows errors next to). */
export function fieldPaths(doc: SettingsDoc): string[] {
  const paths: string[] = [];
  for (const section of SECTION_IDS) {
    const editable = doc[section].editable;
    for (const field of Object.keys(editable)) {
      if (editable[field] && !(section === 'assistant' && field === 'providers')) paths.push(`${section}.${field}`);
    }
  }
  for (const [name, provider] of Object.entries(doc.assistant.providers)) {
    for (const field of Object.keys(provider.editable)) {
      if (provider.editable[field]) paths.push(`assistant.providers.${name}.${field}`);
    }
  }
  return paths;
}

/** Errors no field or section shows: the save bar lists them. */
export function unplacedErrors(doc: SettingsDoc, errors: Record<string, string[]>): string[] {
  const placed = new Set<string>([...fieldPaths(doc), ...SECTION_IDS]);
  return Object.entries(errors)
    .filter(([path]) => !placed.has(path))
    .flatMap(([path, msgs]) => msgs.map((m) => (path ? `${path}: ${m}` : m)));
}

const SOURCE_LABELS: Record<SettingSource, string> = { override: 'override', env: 'from .env', default: 'default' };
export const sourceLabel = (source: SettingSource | undefined) => (source ? SOURCE_LABELS[source] : '');

/** A directory size: "0 B" when empty, "826 KB", "2.0 MB"; "≥ 1.2 GB" when the walk stopped early. */
export function formatStorage({ bytes, truncated }: Pick<SettingsStorage, 'bytes' | 'truncated'>): string {
  let size: string;
  if (bytes <= 0) size = '0 B';
  else if (bytes < 1024 ** 2) size = `${Math.max(1, Math.round(bytes / 1024))} KB`;
  else if (bytes < 1024 ** 3) size = `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  else size = `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return truncated ? `≥ ${size}` : size;
}

/** `order` with the item at `index` moved by `by` (−1 up, +1 down); out of range leaves it. */
export function move<T>(order: T[], index: number, by: -1 | 1): T[] {
  const to = index + by;
  if (index < 0 || index >= order.length || to < 0 || to >= order.length) return order;
  const next = [...order];
  [next[index], next[to]] = [next[to], next[index]];
  return next;
}

export const PROVIDER_LABELS: Record<string, string> = {
  gemini: 'Gemini',
  groq: 'Groq',
  ollama: 'Ollama',
  apertus: 'Apertus',
  anthropic: 'Anthropic',
  mistral: 'Mistral',
  openai: 'OpenAI',
};
export const providerLabel = (name: string) => PROVIDER_LABELS[name] ?? name;

export const DOWNLOAD_POLICY_LABELS: Record<string, string> = {
  all: 'Everyone, guests too',
  members: 'Members and operators',
  operators: 'Operators only',
  none: 'Nobody',
};
