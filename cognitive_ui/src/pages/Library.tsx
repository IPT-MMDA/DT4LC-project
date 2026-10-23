import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { clsx as cn } from 'clsx';
import { resolveApiUrl } from '../api/client';
import { useSession } from '../api/hooks/useSession';
import { FILES_QUERY_KEY, useListFiles, useUploadFile } from '../api/hooks/useUpload';
import { downloadUrl, JOBS_QUERY_KEY, listJobs, statisticsCsvUrl } from '../api/library';
import { StatusPill } from '../components/chat/analysis/StatusPill';
import {
  DEFAULT_FILTERS,
  filterRows,
  libraryRows,
  rowWhen,
  SINCE_LABELS,
  typeOptions,
  type LibraryFilters,
  type LibraryRow,
  type Show,
  type Since,
  type StatusFilter,
} from '../components/library/libraryModel';
import { useLibraryActions } from '../components/library/useLibraryActions';
import { useChatStore } from '../store/useChatStore';

// /app/library (design/screens/library.png): every analysis and file from this
// session in one table, newest first, with filters and one action per row. Replaces
// v0.5's Jobs and Data pages. Classes join with clsx: utils/cn's tailwind-merge
// doesn't know the type-scale tokens and drops `text-control` before a text colour.

const GEOTIFF = /\.tiff?$/i;
const outline = 'px-[10px] py-[5px] rounded-control border border-border bg-surface text-text text-control hover:bg-surface2';
const chip = 'px-[11px] py-[5px] rounded-pill border text-control';
const menuButton = 'px-[10px] py-[5px] rounded-control border border-border bg-surface text-text2 text-control hover:bg-surface2';

/** A dropdown of exclusive choices: "Type: any ▾", "Last 7 days ▾". */
function ChoiceMenu<T extends string | null>({
  label,
  name,
  value,
  options,
  onChange,
}: {
  label: string;
  name: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    const onPointer = (e: PointerEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointer);
    };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <button type="button" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)} className={menuButton}>
        {label} ▾
      </button>
      {open && (
        <div role="menu" aria-label={name} className="absolute left-0 top-full mt-1 z-20 min-w-[180px] bg-surface border border-border rounded-card p-1 shadow-composer">
          {options.map((o) => (
            <button
              key={o.label}
              type="button"
              role="menuitemradio"
              aria-checked={o.value === value}
              onClick={() => {
                onChange(o.value);
                setOpen(false);
              }}
              className={cn('w-full text-left px-2 py-[6px] rounded-control text-control hover:bg-surface2', o.value === value && 'font-semibold')}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** ⋯ of a row: Delete, then a confirm in place. */
function RowMenu({ row, onDelete }: { row: LibraryRow; onDelete: () => Promise<void> }) {
  const [state, setState] = useState<'closed' | 'menu' | 'confirm' | 'deleting'>('closed');
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (state === 'closed') return;
    const onPointer = (e: PointerEvent) => ref.current && !ref.current.contains(e.target as Node) && setState('closed');
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setState('closed');
    document.addEventListener('pointerdown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [state]);
  const what = row.kind === 'analysis' ? 'this analysis and its files' : row.files.length > 1 ? 'both files' : 'this file';
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-label={`More for ${row.title}`}
        aria-haspopup="menu"
        aria-expanded={state !== 'closed'}
        onClick={() => setState((s) => (s === 'closed' ? 'menu' : 'closed'))}
        className="w-7 h-7 rounded-control text-muted hover:bg-surface2"
      >
        ⋯
      </button>
      {state === 'menu' && (
        <div role="menu" aria-label={`Actions for ${row.title}`} className="absolute right-0 top-full mt-1 z-20 w-[160px] bg-surface border border-border rounded-card p-1 shadow-composer">
          <button type="button" role="menuitem" onClick={() => setState('confirm')} className="w-full text-left px-2 py-[6px] rounded-control text-control text-fail-ink hover:bg-surface2">
            Delete…
          </button>
        </div>
      )}
      {(state === 'confirm' || state === 'deleting') && (
        <div role="alertdialog" aria-label={`Delete ${row.title}?`} className="absolute right-0 top-full mt-1 z-20 w-[240px] bg-surface border border-border rounded-card p-3 shadow-composer text-left">
          <p className="text-control text-text">Delete {what}? This can't be undone.</p>
          <div className="mt-2 flex justify-end gap-2">
            <button type="button" onClick={() => setState('closed')} className={outline}>
              Keep
            </button>
            <button
              type="button"
              disabled={state === 'deleting'}
              onClick={async () => {
                setState('deleting');
                await onDelete();
                setState('closed');
              }}
              className="px-[10px] py-[5px] rounded-control bg-fail text-on-accent text-control font-medium disabled:opacity-50"
            >
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Thumbnail({ url }: { url: string | null }) {
  const [broken, setBroken] = useState(false);
  if (!url || broken) return <div data-testid="library-thumbnail" aria-hidden className="w-[56px] h-[38px] rounded-chip border border-border bg-surface2" />;
  return (
    <img
      data-testid="library-thumbnail"
      src={resolveApiUrl(url)}
      alt=""
      onError={() => setBroken(true)}
      className="w-[56px] h-[38px] rounded-chip border border-border object-cover bg-surface2"
    />
  );
}

function RowActionButton({ row, onAction }: { row: LibraryRow; onAction: () => void }) {
  const stop = (e: MouseEvent) => e.stopPropagation();
  switch (row.action) {
    case 'csv':
      return row.kind === 'analysis' ? (
        <a href={statisticsCsvUrl(row.job.id)} download onClick={stop} className={outline}>
          CSV
        </a>
      ) : null;
    case 'download':
      return row.kind === 'file' ? (
        <a
          href={downloadUrl(row.files[0].path, row.files[0].filename)}
          download={row.files[0].filename}
          title={row.files[0].filename}
          onClick={stop}
          aria-label={`Download ${row.title}`}
          className={outline}
        >
          ↓
        </a>
      ) : null;
    case 'map':
    case 'chat':
    case 'retry':
      return (
        <button
          type="button"
          onClick={(e) => {
            stop(e);
            onAction();
          }}
          className={outline}
        >
          {{ map: 'Map', chat: 'Chat', retry: 'Retry' }[row.action]}
        </button>
      );
    default:
      return null;
  }
}

export function Library() {
  const queryClient = useQueryClient();
  const { data: session } = useSession();
  // Both refetch on every visit: analyses and files are added from the chat.
  const jobs = useQuery({ queryKey: JOBS_QUERY_KEY, queryFn: listJobs, staleTime: 0 });
  const files = useListFiles({ fresh: true });
  const conversations = useChatStore((s) => s.conversations);
  const actions = useLibraryActions();
  const upload = useUploadFile();
  const fileInput = useRef<HTMLInputElement>(null);
  const [filters, setFilters] = useState<LibraryFilters>(DEFAULT_FILTERS);
  const [error, setError] = useState<string | null>(null);
  const now = Date.now();

  const rows = useMemo(() => libraryRows(jobs.data ?? [], files.data ?? [], conversations), [jobs.data, files.data, conversations]);
  const shown = filterRows(rows, filters, now);
  const types = typeOptions(rows);
  const loading = jobs.isPending || files.isPending;
  const failedToLoad = jobs.isError || files.isError;

  const set = (patch: Partial<LibraryFilters>) => setFilters((f) => ({ ...f, ...patch }));
  const toggleStatus = (s: StatusFilter) =>
    setFilters((f) => ({ ...f, statuses: f.statuses.includes(s) ? f.statuses.filter((x) => x !== s) : [...f.statuses, s] }));
  const run = async (action: () => Promise<string | null> | string | null) => {
    setError(null);
    setError(await action());
  };

  const onUpload = async (list: FileList | null) => {
    const picked = [...(list ?? [])];
    if (fileInput.current) fileInput.current.value = '';
    const rejected = picked.filter((f) => !GEOTIFF.test(f.name));
    setError(rejected.length > 0 ? `Not a GeoTIFF: ${rejected.map((f) => f.name).join(', ')}` : null);
    for (const file of picked.filter((f) => GEOTIFF.test(f.name))) {
      try {
        await upload.mutateAsync(file);
      } catch (err) {
        setError(`Upload failed: ${file.name}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    await queryClient.invalidateQueries({ queryKey: FILES_QUERY_KEY });
  };

  const intro = 'Every analysis and file from your conversations.';
  const subtitle = session?.kept_hours != null ? `${intro} Guest items expire after ${session.kept_hours} h.` : intro;
  const segment = (value: Show, label: string) => (
    <button
      type="button"
      aria-pressed={filters.show === value}
      onClick={() => set({ show: value })}
      className={cn('px-[13px] py-[5px] text-control border-l border-border first:border-l-0', filters.show === value ? 'bg-ink text-ink-bg font-semibold' : 'bg-surface text-text2 hover:bg-surface2')}
    >
      {label}
    </button>
  );
  const statusChip = (value: StatusFilter, label: string) => {
    const on = filters.statuses.includes(value);
    return (
      <button type="button" aria-pressed={on} onClick={() => toggleStatus(value)} className={cn(chip, on ? 'bg-ink text-ink-bg border-ink font-semibold' : 'bg-surface border-border text-text2 hover:bg-surface2')}>
        {label}
      </button>
    );
  };

  let body: ReactNode;
  if (loading) {
    body = <p className="px-4 py-6 text-control text-muted">Loading…</p>;
  } else if (failedToLoad) {
    body = (
      <p role="alert" className="px-4 py-6 text-control text-fail-ink">
        Couldn't load the library.{' '}
        <button type="button" onClick={() => void Promise.all([jobs.refetch(), files.refetch()])} className="underline-offset-2 hover:underline">
          Try again
        </button>
      </p>
    );
  } else if (rows.length === 0) {
    body = (
      <div data-testid="library-empty" className="px-4 py-10 text-center">
        <p className="text-card">Nothing here yet</p>
        <p className="mt-1 text-body text-muted">Analyses and files from your conversations will collect here.</p>
        <Link to="/app" className="mt-3 inline-block text-control font-medium text-accent-ink">
          Go to Chat
        </Link>
      </div>
    );
  } else if (shown.length === 0) {
    body = (
      <div data-testid="library-filtered-empty" className="px-4 py-10 text-center">
        <p className="text-body text-muted">Nothing matches these filters.</p>
        <button type="button" onClick={() => setFilters({ ...DEFAULT_FILTERS, since: 'all' })} className="mt-2 text-control font-medium text-accent-ink">
          Clear filters
        </button>
      </div>
    );
  } else {
    const head = 'px-3 py-[10px] text-left text-label uppercase text-muted';
    body = (
      <table className="w-full table-fixed border-collapse">
        <thead>
          <tr>
            <th scope="col" className={cn(head, 'w-[88px]')}>
              <span className="sr-only">Preview</span>
            </th>
            <th scope="col" className={head}>Name</th>
            <th scope="col" className={cn(head, 'w-[15%]')}>Type</th>
            <th scope="col" className={cn(head, 'w-[13%]')}>Status</th>
            <th scope="col" className={cn(head, 'w-[13%]')}>Conversation</th>
            <th scope="col" className={cn(head, 'w-[10%]')}>When</th>
            <th scope="col" className={cn(head, 'w-[104px]')}>
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {shown.map((row) => (
            <tr
              key={row.key}
              data-testid="library-row"
              data-kind={row.kind}
              data-id={row.kind === 'analysis' ? row.job.id : (row.deleteId ?? row.key)}
              onClick={() => void run(() => actions.open(row))}
              className="border-t border-border2 cursor-pointer hover:bg-bg"
            >
              <td className="pl-4 pr-2 py-[10px]">
                <Thumbnail url={row.thumbnail} />
              </td>
              <td className="px-3 py-[10px] min-w-0">
                <button type="button" className="block max-w-full text-left">
                  <span data-testid="library-name" className="block text-control font-medium text-text truncate">{row.title}</span>
                  {row.subtitle && <span data-testid="library-subtitle" className="block text-[11.5px] text-muted truncate">{row.subtitle}</span>}
                </button>
              </td>
              <td data-testid="library-type" className="px-3 py-[10px] text-control text-text2 truncate">{row.type || '—'}</td>
              <td data-testid="library-status" className="px-3 py-[10px]">
                {row.kind === 'analysis' ? <StatusPill status={row.status} detail={row.duration} /> : <span className="text-control text-muted">—</span>}
              </td>
              <td data-testid="library-conversation" className="px-3 py-[10px] text-control text-text2 truncate">{row.conversation}</td>
              <td data-testid="library-when" className="px-3 py-[10px] font-mono text-[11.5px] text-muted whitespace-nowrap">{rowWhen(row, now)}</td>
              <td className="pr-3 py-[10px]" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-end gap-1">
                  <RowActionButton
                    row={row}
                    onAction={() =>
                      void run(() =>
                        row.kind === 'file' ? actions.openInChat(row) : row.action === 'map' ? actions.openOnMap(row) : actions.retry(row),
                      )
                    }
                  />
                  {(row.kind === 'analysis' || row.deleteId) && <RowMenu row={row} onDelete={() => run(() => actions.remove(row))} />}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  return (
    <main data-testid="library-page" className="flex-1 min-w-0 overflow-y-auto bg-bg px-[40px] py-[32px]">
      <header className="flex items-start gap-4">
        <div className="flex-1 min-w-0">
          <h1 className="text-title">Library</h1>
          <p className="mt-1 text-body text-text2">{subtitle}</p>
        </div>
        <input ref={fileInput} type="file" accept=".tif,.tiff" multiple hidden aria-label="GeoTIFF file" onChange={(e) => void onUpload(e.target.files)} />
        <button type="button" disabled={upload.isPending} onClick={() => fileInput.current?.click()} className="px-[14px] py-[8px] rounded-control border border-border bg-surface text-control font-medium hover:bg-surface2 disabled:opacity-50">
          {upload.isPending ? 'Uploading…' : 'Upload GeoTIFF'}
        </button>
      </header>

      <div role="toolbar" aria-label="Filters" className="mt-6 flex items-center flex-wrap gap-2">
        <div role="group" aria-label="Show" className="flex rounded-control border border-border overflow-hidden">
          {segment('all', 'All')}
          {segment('analyses', 'Analyses')}
          {segment('files', 'Files')}
        </div>
        {statusChip('done', 'Done')}
        {statusChip('failed', 'Failed')}
        {statusChip('cancelled', 'Cancelled')}
        <ChoiceMenu
          name="Type"
          label={`Type: ${filters.type ?? 'any'}`}
          value={filters.type}
          options={[{ value: null, label: 'Any type' }, ...types.map((t) => ({ value: t, label: t }))]}
          onChange={(type) => set({ type })}
        />
        <ChoiceMenu<Since>
          name="When"
          label={SINCE_LABELS[filters.since]}
          value={filters.since}
          options={(Object.keys(SINCE_LABELS) as Since[]).map((s) => ({ value: s, label: SINCE_LABELS[s] }))}
          onChange={(since) => set({ since })}
        />
        <div className="flex-1" />
        <span data-testid="library-count" className="text-control text-muted">
          {shown.length === 1 ? '1 item' : `${shown.length} items`}
        </span>
      </div>

      {error && (
        <p role="alert" className="mt-3 text-control text-fail-ink">
          {error}
        </p>
      )}

      <section aria-label="Library items" className="mt-4 bg-surface border border-border rounded-card overflow-visible">
        {body}
      </section>
    </main>
  );
}
