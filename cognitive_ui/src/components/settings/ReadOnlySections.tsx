import type { SettingsDoc } from '../../api/settings';
import { InfoRow, Status } from './fields';
import { formatStorage } from './settingsModel';

const box = 'border border-border rounded-card bg-surface overflow-hidden text-control';
const yesNo = (ok: boolean, yes: string, no: string) => (ok ? <Status hue="done">{yes}</Status> : <Status hue="muted">{no}</Status>);

/** Data sources: Earth Engine credentials, the uploads directory, the bundled sample. Read-only. */
export function DataSourcesSection({ doc }: { doc: SettingsDoc }) {
  const { earth_engine: ee, uploads, sample } = doc.data_sources;
  return (
    <div className={box}>
      <InfoRow label="Earth Engine">
        <span data-testid="ee-status">{yesNo(ee.configured, 'Configured', 'Not configured')}</span>
        <span className="text-[12px] text-muted">
          Project id {ee.project_id_present ? 'present' : 'missing'} · key file {ee.key_file_present ? 'present' : 'missing'}
        </span>
      </InfoRow>
      <InfoRow label="Uploads">
        <span className="font-mono text-[12px] break-all">{uploads.path}</span>
        <span className="text-[12px] text-muted">
          {uploads.files} {uploads.files === 1 ? 'file' : 'files'} · {formatStorage({ bytes: uploads.bytes, truncated: false })}
        </span>
      </InfoRow>
      <InfoRow label="Sample">
        <span className="font-mono text-[12px]">{sample.name}</span>
        {yesNo(sample.bundled, 'Bundled', 'Not bundled')}
        <span className="font-mono text-[12px] text-muted break-all">{sample.path}</span>
      </InfoRow>
    </div>
  );
}

const DEPLOYMENT = { local: 'Local server', demo: 'Public demo' } as const;

/** Server: version, deployment, paths with their storage, job counts. Read-only. */
export function ServerSection({ doc }: { doc: SettingsDoc }) {
  const { server } = doc;
  const truncated = Object.values(server.storage).some((s) => s.truncated);
  const counts = Object.entries(server.jobs.by_status);
  return (
    <div className="flex flex-col gap-5">
      <div className={box}>
        <InfoRow label="Version">
          <span className="font-mono">{server.version}</span>
        </InfoRow>
        <InfoRow label="Deployment">{DEPLOYMENT[server.deployment] ?? server.deployment}</InfoRow>
        <InfoRow label="Jobs">
          <span>
            {server.jobs.total} in total · {server.jobs.queue_size} queued · {server.jobs.workers} workers
          </span>
          <span className="text-[12px] text-muted">{counts.map(([status, n]) => `${status} ${n}`).join(' · ')}</span>
        </InfoRow>
      </div>

      <table aria-label="Storage" className="w-full border border-border rounded-card bg-surface text-control border-separate border-spacing-0 overflow-hidden">
        <thead>
          <tr className="text-left text-label uppercase text-muted">
            <th scope="col" className="px-4 py-2 font-semibold border-b border-border2">Directory</th>
            <th scope="col" className="px-4 py-2 font-semibold border-b border-border2">Path</th>
            <th scope="col" className="px-4 py-2 font-semibold border-b border-border2 text-right">Files</th>
            <th scope="col" className="px-4 py-2 font-semibold border-b border-border2 text-right">Size</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(server.paths).map(([key, dir]) => {
            const usage = server.storage[key];
            return (
              <tr key={key} data-testid="storage-row" data-key={key}>
                <th scope="row" className="px-4 py-[9px] text-left font-medium border-b border-border2 capitalize">
                  {key}
                </th>
                <td className="px-4 py-[9px] font-mono text-[12px] text-text2 border-b border-border2 break-all">{dir}</td>
                <td className="px-4 py-[9px] font-mono text-[12px] text-right border-b border-border2">
                  {usage ? `${usage.truncated ? '≥ ' : ''}${usage.files}` : ''}
                </td>
                <td className="px-4 py-[9px] font-mono text-[12px] text-right border-b border-border2 whitespace-nowrap">
                  {usage ? formatStorage(usage) : ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-[12px] text-muted">
        The cache holds the other directories unless they are set elsewhere, so sizes overlap.
        {truncated && ' ≥: the count stopped early on a very large directory; the real size is larger.'}
      </p>
    </div>
  );
}
