import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { SESSION_QUERY_KEY } from '../api/hooks/useSession';
import { SETTINGS_QUERY_KEY, getSettings, putSettings, type SettingsDoc } from '../api/settings';
import { AccessSection } from '../components/settings/AccessSection';
import { AssistantSection } from '../components/settings/AssistantSection';
import { makeForm, smallButton } from '../components/settings/form';
import { ModelsSection } from '../components/settings/ModelsSection';
import { DataSourcesSection, ServerSection } from '../components/settings/ReadOnlySections';
import {
  SECTION_IDS,
  SECTION_LABELS,
  buildUpdate,
  dirtySections,
  errorsByField,
  resetEdit,
  setEdit,
  undoEdit,
  unplacedErrors,
  type Edits,
  type SectionId,
} from '../components/settings/settingsModel';
import { cn } from '../utils/cn';

const SUBTITLES: Record<SectionId, string> = {
  assistant: 'Which providers answer, in which order, with which models.',
  models: 'Model weights on this server and who may download them.',
  access: 'Who signs in as a member or an operator, and what guests get.',
  data_sources: 'Where imagery comes from. Read-only: set in the server’s environment.',
  server: 'This installation. Read-only.',
};

/** The five sections of GET /v1/settings with a save bar; exported for tests. */
export function SettingsView({ doc }: { doc: SettingsDoc }) {
  const queryClient = useQueryClient();
  const [section, setSection] = useState<SectionId>('assistant');
  const [edits, setEdits] = useState<Edits>({});
  const [errors, setErrors] = useState<Record<string, string[]>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const change = (next: (edits: Edits) => Edits, path: string) => {
    setEdits(next);
    setSaved(false);
    setErrors((current) => {
      if (!current[path]) return current;
      const rest = { ...current };
      delete rest[path];
      return rest;
    });
  };
  const form = makeForm(doc, edits, errors, {
    set: (path, value) => change((e) => setEdit(doc, e, path, value), path),
    reset: (path) => change((e) => resetEdit(e, path), path),
    undo: (path) => change((e) => undoEdit(e, path), path),
  });

  const dirty = dirtySections(edits);
  const count = Object.keys(edits).length;
  const general = unplacedErrors(doc, errors);

  const save = async () => {
    setSaving(true);
    setSaved(false);
    try {
      const next = await putSettings(buildUpdate(edits));
      queryClient.setQueryData(SETTINGS_QUERY_KEY, next);
      setEdits({});
      setErrors({});
      setSaved(true);
      await Promise.all(
        [SETTINGS_QUERY_KEY, ['capabilities'], ['health'], SESSION_QUERY_KEY].map((queryKey) =>
          queryClient.invalidateQueries({ queryKey }),
        ),
      );
    } catch (err) {
      const found =
        err instanceof ApiError && err.status === 422
          ? errorsByField(err.body)
          : { '': [err instanceof Error ? err.message : 'Not saved'] };
      setErrors(found);
      // If the open section has no error, open the first one that has.
      const failed = new Set(Object.keys(found).map((p) => p.split('.')[0]));
      const first = SECTION_IDS.find((id) => failed.has(id));
      if (first && !failed.has(section)) setSection(first);
    } finally {
      setSaving(false);
    }
  };
  const discard = () => {
    setEdits({});
    setErrors({});
    setSaved(false);
  };

  const errorSections = new Set(Object.keys(errors).map((p) => p.split('.')[0]));

  return (
    <>
      <header>
        <h1 id="settings-title" className="text-title tracking-[-0.01em]">
          Settings · this server
        </h1>
        <p className="mt-1 max-w-[720px] text-[13px] leading-[1.5] text-muted">
          Operator settings for this installation. Saved values are overrides kept on the server and apply at once; the
          .env file is never written.
        </p>
      </header>

      <div className="grid grid-cols-[180px_1fr] gap-8 items-start">
        <nav aria-label="Settings sections" className="sticky top-0 flex flex-col gap-[2px]">
          {SECTION_IDS.map((id) => (
            <button
              key={id}
              type="button"
              aria-current={section === id ? 'true' : undefined}
              data-testid={`settings-tab-${id}`}
              onClick={() => setSection(id)}
              className={cn(
                'px-3 py-[7px] rounded-[8px] text-left text-control flex items-center justify-between gap-2',
                section === id ? 'bg-accent-tint text-accent-ink font-semibold' : 'text-text2 hover:bg-surface2',
              )}
            >
              {SECTION_LABELS[id]}
              {errorSections.has(id) ? (
                <span aria-label="has errors" className="w-[6px] h-[6px] rounded-full bg-fail" />
              ) : (
                dirty.has(id) && <span aria-label="unsaved changes" className="w-[6px] h-[6px] rounded-full bg-accent" />
              )}
            </button>
          ))}
        </nav>

        <div className="min-w-0 flex flex-col gap-4">
          <div
            data-testid="settings-savebar"
            data-dirty={count > 0 ? 'true' : 'false'}
            className={cn(
              'sticky top-0 z-10 flex items-center gap-3 px-4 py-[9px] rounded-card border text-control',
              count > 0 ? 'bg-accent-tint border-accent-tint text-accent-ink' : 'bg-surface border-border text-muted',
            )}
          >
            <span role="status" className="flex-1">
              {count > 0
                ? `Unsaved changes: ${count} ${count === 1 ? 'field' : 'fields'} in ${[...dirty].map((id) => SECTION_LABELS[id as SectionId] ?? id).join(', ')}`
                : saved
                  ? 'Saved.'
                  : 'No unsaved changes.'}
            </span>
            <button type="button" className={smallButton} disabled={count === 0 || saving} onClick={discard}>
              Discard
            </button>
            <button
              type="button"
              disabled={count === 0 || saving}
              onClick={() => void save()}
              className="px-[12px] py-[4px] rounded-control text-[12px] font-semibold bg-accent text-on-accent disabled:bg-surface2 disabled:text-faint disabled:cursor-not-allowed"
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
          {general.length > 0 && (
            <div role="alert" data-testid="settings-errors" className="px-4 py-[9px] rounded-card bg-fail-tint text-fail-ink text-control">
              Not saved. {general.join(' ')}
            </div>
          )}

          <section aria-labelledby={`settings-${section}-title`} data-testid={`settings-section-${section}`} className="flex flex-col gap-4">
            <div className="flex items-baseline gap-3 flex-wrap">
              <h2 id={`settings-${section}-title`} className="text-[16px] font-semibold">
                {SECTION_LABELS[section]}
              </h2>
              <span className="text-[13px] text-muted">{SUBTITLES[section]}</span>
            </div>
            {errors[section]?.map((msg) => (
              <p key={msg} role="alert" data-testid="field-error" className="text-[12px] text-fail-ink">
                {msg}
              </p>
            ))}
            {section === 'assistant' && <AssistantSection form={form} />}
            {section === 'models' && <ModelsSection form={form} />}
            {section === 'access' && <AccessSection form={form} />}
            {section === 'data_sources' && <DataSourcesSection doc={doc} />}
            {section === 'server' && <ServerSection doc={doc} />}
          </section>
        </div>
      </div>
    </>
  );
}

/** /app/settings (operators; the route shows the not-found page to anyone else). */
export function Settings() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: SETTINGS_QUERY_KEY, queryFn: getSettings });
  return (
    <main
      data-testid="settings-page"
      aria-labelledby="settings-title"
      className="flex-1 min-w-0 h-full overflow-y-auto bg-bg px-10 pt-8 pb-12 flex flex-col gap-[30px] text-[13px]"
    >
      {data ? (
        <SettingsView doc={data} />
      ) : (
        <>
          <h1 id="settings-title" className="text-title tracking-[-0.01em]">
            Settings · this server
          </h1>
          {isPending && <p className="text-muted">Reading the server’s settings…</p>}
          {isError && (
            <p className="text-fail-ink">
              Could not read the settings.{' '}
              <button type="button" onClick={() => void refetch()} className="underline">
                Try again
              </button>
            </p>
          )}
        </>
      )}
    </main>
  );
}
