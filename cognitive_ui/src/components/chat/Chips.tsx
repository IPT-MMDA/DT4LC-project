import { resolveApiUrl } from '../../api/client';
import type { ChatRegion } from '../../types';
import { cn } from '../../utils/cn';
import type { AttachmentChip } from './attachmentChips';
import { regionChipLines } from './regionChip';

// Chips: a thumbnail, a mono name and a muted line. In the composer they
// have a remove ×; on sent messages they are read-only (no onRemove).

type Size = 'md' | 'sm';

function RemoveButton({ label, onRemove }: { label: string; onRemove: () => void }) {
  return (
    <button
      type="button"
      onClick={onRemove}
      aria-label={`Remove ${label}`}
      className="text-[12px] leading-none text-muted hover:text-text px-1 py-1 rounded-chip"
    >
      ×
    </button>
  );
}

export function FileChip({ chip, onRemove, size = 'sm' }: { chip: AttachmentChip; onRemove?: () => void; size?: Size }) {
  // The later file's preview: for the sample pair, 9 June (after the breach).
  const preview = [...chip.attachments].reverse().find((a) => a.preview_url)?.preview_url;
  const thumb = size === 'md' ? 'w-9 h-9 rounded-[5px]' : 'w-[30px] h-[30px] rounded-chip';
  return (
    <div
      data-testid="attachment-chip"
      className={cn(
        'inline-flex items-center gap-[10px] border border-border bg-surface max-w-full',
        size === 'md' ? 'pl-[6px] pr-3 py-[6px] rounded-[8px]' : 'pl-1 pr-2 py-1 rounded-[7px]',
      )}
    >
      {preview ? (
        <img src={resolveApiUrl(preview)} alt="" className={cn(thumb, 'object-cover flex-none')} />
      ) : (
        <div aria-hidden className={cn(thumb, 'bg-surface2 flex-none')} />
      )}
      <div className="min-w-0">
        <div className={cn('font-mono truncate', size === 'md' ? 'text-[12.5px]' : 'text-[12px]')}>{chip.name}</div>
        {chip.line && <div className="text-[11px] text-muted truncate">{chip.line}</div>}
      </div>
      {onRemove && <RemoveButton label={chip.name} onRemove={onRemove} />}
    </div>
  );
}

export function RegionChip({ region, onRemove }: { region: ChatRegion; onRemove?: () => void }) {
  const { title, detail } = regionChipLines(region);
  return (
    <div
      data-testid="region-chip"
      className="inline-flex items-center gap-2 border border-border bg-bg rounded-[7px] pl-1 pr-2 py-1 max-w-full"
    >
      <div aria-hidden className="w-[26px] h-[26px] rounded-chip bg-surface2 flex-none relative">
        <div className="absolute inset-[6px_5px_5px_7px] border-[1.5px] border-canc" />
      </div>
      <div className="min-w-0">
        <div className="text-[11.5px] font-mono truncate">{title}</div>
        <div className="text-[10px] text-muted truncate">{detail}</div>
      </div>
      {onRemove && <RemoveButton label="region" onRemove={onRemove} />}
    </div>
  );
}
