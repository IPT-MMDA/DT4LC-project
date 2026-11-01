import { useEffect, useId, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { SETTINGS_QUERY_KEY, getModelStatus, preDownloadModel, type SettingsModelItem } from '../../api/settings';
import { formatDuration } from '../chat/analysis/format';
import { formatBytes } from '../detail/resultDetailModel';
import { cn } from '../../utils/cn';
import { FieldRow, Status } from './fields';
import { inputClass, smallButton, type SettingsForm } from './form';
import { DOWNLOAD_POLICY_LABELS } from './settingsModel';

const POLL_MS = 1000;

/**
 * One model: install state, size, licence, and Pre-download while it is not installed. A
 * started download is followed with GET /v1/ml-models/{id} every second until it is installed
 * (then the settings and capabilities are fetched again) or fails.
 */
function ModelRow({ item, path }: { item: SettingsModelItem; path: string }) {
  const queryClient = useQueryClient();
  const [started, setStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const status = useQuery({
    queryKey: ['ml-model', item.id],
    queryFn: () => getModelStatus(item.id),
    enabled: started,
    refetchInterval: (query) => (query.state.data?.status === 'downloading' || !query.state.data ? POLL_MS : false),
  });
  const state = started ? status.data?.status : undefined;
  const installed = item.installed || state === 'available';

  useEffect(() => {
    if (state === 'available') {
      void queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEY });
      void queryClient.invalidateQueries({ queryKey: ['capabilities'] });
    }
    if (state === 'failed') setError(status.data?.error || 'The download failed');
  }, [state, status.data?.error, queryClient]);

  const start = async () => {
    setError(null);
    try {
      await preDownloadModel(path, item.id);
      setStarted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start the download');
    }
  };

  const downloading = started && !installed && state !== 'failed';
  const percent = status.data?.download_progress?.percent;
  return (
    <div
      data-testid="settings-model"
      data-id={item.id}
      className="grid grid-cols-[1.3fr_1fr_90px_120px_130px] gap-[14px] px-4 py-[10px] border-b border-border2 last:border-b-0 items-center"
    >
      <span className="font-medium">{item.display_name}</span>
      <span>
        {installed ? (
          <Status hue="done">Installed</Status>
        ) : downloading ? (
          <Status hue="run">{percent === undefined ? 'Downloading…' : `Downloading · ${Math.round(percent)} %`}</Status>
        ) : (
          <Status hue="muted">Not installed · about {formatDuration(item.est_download_s)}</Status>
        )}
      </span>
      <span className="font-mono text-[12px] text-text2">{formatBytes(item.size_bytes)}</span>
      <a href={item.licence_url} target="_blank" rel="noreferrer" className="font-mono text-[12px] text-muted hover:text-accent-ink hover:underline">
        {item.licence}
      </a>
      <span className="justify-self-end">
        {!installed && (
          <button type="button" aria-label={`Pre-download ${item.display_name}`} className={smallButton} disabled={downloading} onClick={() => void start()}>
            {downloading ? 'Downloading…' : 'Pre-download'}
          </button>
        )}
      </span>
      {error && (
        <p role="alert" className="col-span-full text-[12px] text-fail-ink">
          {error}
        </p>
      )}
    </div>
  );
}

/** Models: install state per model with Pre-download, and who may download on first use. */
export function ModelsSection({ form }: { form: SettingsForm }) {
  const { models } = form.doc;
  const path = 'models.download_policy';
  const selectId = useId();
  return (
    <div className="flex flex-col gap-5">
      <div className="border border-border rounded-card bg-surface overflow-hidden text-control">
        {models.items.map((item) => (
          <ModelRow key={item.id} item={item} path={models.pre_download.path} />
        ))}
      </div>
      {form.field(path).editable ? (
        <FieldRow
          form={form}
          path={path}
          label="Downloads on first use"
          labelFor={selectId}
          hint="Who may make the server download a model the first time an analysis needs it. Pre-download works for operators unless this is Nobody."
        >
          {({ describedBy, resetting }) => (
            <select
              id={selectId}
              value={form.value<string>(path)}
              disabled={resetting}
              aria-describedby={describedBy}
              onChange={(e) => form.set(path, e.target.value)}
              className={cn(inputClass, 'self-start')}
            >
              {models.download_policy_options.map((option) => (
                <option key={option} value={option}>
                  {DOWNLOAD_POLICY_LABELS[option] ?? option}
                </option>
              ))}
            </select>
          )}
        </FieldRow>
      ) : (
        <p className="text-control text-text2">Downloads on first use: {DOWNLOAD_POLICY_LABELS[models.download_policy] ?? models.download_policy}</p>
      )}
    </div>
  );
}
