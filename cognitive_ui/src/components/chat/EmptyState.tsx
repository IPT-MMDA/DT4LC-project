import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { attachSample } from '../../hooks/useSample';
import { useChatStore } from '../../store/useChatStore';
import { apiLogger as logger } from '../../utils/logger';
import { attachmentChips } from './attachmentChips';
import { FileChip } from './Chips';
import { SAMPLE_SUGGESTIONS, SUGGESTIONS } from './suggestions';

interface EmptyStateProps {
  /** A suggestion sends its prompt right away (as in the prototype). */
  onSend: (prompt: string) => void;
  onOpenFilePicker: () => void;
  sending: boolean;
}

/**
 * "What do you want to know?" (workspace-empty-*.png): the files already in the
 * composer (with ×) or three ways to add some, then suggested prompts.
 */
export function EmptyState({ onSend, onOpenFilePicker, sending }: EmptyStateProps) {
  const navigate = useNavigate();
  const pending = useChatStore((s) => s.pendingAttachments);
  const removeAttachment = useChatStore((s) => s.removeAttachment);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const chips = attachmentChips(pending);
  const hasSample = pending.some((a) => a.group);
  const suggestions = hasSample ? SAMPLE_SUGGESTIONS : SUGGESTIONS;

  const addSample = () => {
    setSampleError(null);
    attachSample().catch((err: unknown) => {
      logger.warn('Could not attach the sample:', err);
      setSampleError("The sample isn't available on this server.");
    });
  };

  const action = 'px-3 py-2 border border-dashed border-border rounded-[8px] text-control text-text2 text-left hover:bg-surface';

  return (
    <section className="my-auto px-6 py-5 flex flex-col gap-[22px]" aria-labelledby="empty-title">
      <h2 id="empty-title" className="text-[24px] font-semibold leading-tight tracking-[-0.015em]">
        {chips.length > 0 ? 'What do you want to know about this area?' : 'What do you want to know?'}
      </h2>
      <p className="text-[13.5px] leading-[1.55] text-text2">
        {hasSample
          ? 'The Kakhovka sample is attached. Describe an analysis in your own words, or pick one to see how it works.'
          : chips.length > 0
            ? 'Your files are attached. Describe an analysis in your own words, or pick one to see how it works.'
            : 'Attach a GeoTIFF, pick a region on the map, or start with the Kakhovka sample. Then describe the analysis in your own words.'}
      </p>

      {chips.length > 0 ? (
        <ul aria-label="Attachments" className="flex flex-col items-start gap-2">
          {chips.map((chip) => (
            <li key={chip.key} className="max-w-full">
              <FileChip
                chip={chip}
                size="md"
                onRemove={() => chip.attachments.forEach((a) => removeAttachment(a.id))}
              />
            </li>
          ))}
        </ul>
      ) : (
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            <button type="button" onClick={addSample} className={action}>
              + Attach the Kakhovka sample
            </button>
            <button type="button" onClick={onOpenFilePicker} className={action}>
              Upload GeoTIFF
            </button>
            <button type="button" onClick={() => navigate('/app/map')} className={action}>
              Pick a region on the map
            </button>
          </div>
          {sampleError && <p className="text-control text-fail-ink">{sampleError}</p>}
        </div>
      )}

      <ul aria-label="Suggestions" className="flex flex-col gap-2">
        {suggestions.map((s) => (
          <li key={s.prompt}>
            <button
              type="button"
              disabled={sending}
              onClick={() => onSend(s.prompt)}
              className="w-full px-[14px] py-3 border border-border rounded-[9px] bg-surface flex justify-between items-center gap-3 text-left text-body hover:border-border2 hover:bg-surface2/40 disabled:cursor-wait"
            >
              <span>{s.prompt}</span>
              <span className="flex items-center gap-[6px] text-[12px] text-muted whitespace-nowrap">
                {s.badge && (
                  <span className="px-[6px] py-[1px] rounded-pill bg-accent-tint text-accent-ink font-semibold text-[10.5px]">
                    {s.badge}
                  </span>
                )}
                {s.time}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
