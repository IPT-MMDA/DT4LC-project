import { useCallback, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react';
import { ArrowUp } from 'lucide-react';
import { useHealth } from '../../api/hooks/useHealth';
import { useQuotaLine } from '../../api/hooks/useSession';
import { useFromMap } from '../../hooks/useFromMap';
import { useChatStore } from '../../store/useChatStore';
import { attachmentChips } from './attachmentChips';
import { FileChip, RegionChip } from './Chips';
import { LibraryPicker } from './LibraryPicker';

const OFFLINE_LINE = 'Assistant offline — analyses still run; answers are limited to the catalog.';
const MAX_INPUT_PX = 160;

interface ComposerProps {
  /** The conversation has messages: "Ask a follow-up…", and file chips show here, not in the empty state. */
  hasMessages: boolean;
  sending: boolean;
  onSend: () => void;
  onOpenFilePicker: () => void;
  /** Uploading or an upload error, shown above the input. */
  note?: string | null;
}

export function Composer({ hasMessages, sending, onSend, onOpenFilePicker, note }: ComposerProps) {
  const draft = useChatStore((s) => s.draft);
  const setDraft = useChatStore((s) => s.setDraft);
  const pending = useChatStore((s) => s.pendingAttachments);
  const region = useChatStore((s) => s.pendingRegion);
  const removeAttachment = useChatStore((s) => s.removeAttachment);
  const setPendingRegion = useChatStore((s) => s.setPendingRegion);
  const offline = useHealth().data?.assistant?.offline === true;
  // A guest with no analysis left: conversation still works and typing is never blocked.
  const quota = useQuotaLine();
  const fromMap = useFromMap();
  const [libraryOpen, setLibraryOpen] = useState(false);
  const libraryRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const closeLibrary = useCallback(() => setLibraryOpen(false), []);

  const chips = hasMessages ? attachmentChips(pending) : [];
  const canSend = !sending && (draft.trim() !== '' || pending.length > 0 || region !== null);

  // Grow with the text up to MAX_INPUT_PX, then scroll.
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_INPUT_PX)}px`;
  }, [draft]);

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== 'Enter' || e.shiftKey || e.nativeEvent.isComposing) return;
    e.preventDefault();
    if (canSend) onSend();
  };

  const control = 'h-7 px-[9px] rounded-control border border-border text-[12px] text-muted hover:text-text hover:bg-surface2';

  return (
    <div className="flex-none px-5 pb-4">
      <div className="bg-surface border border-border rounded-composer shadow-composer pt-[10px] px-3 pb-2 flex flex-col gap-2">
        {(chips.length > 0 || region) && (
          <ul aria-label="Attachments" className="flex flex-wrap gap-2">
            {chips.map((chip) => (
              <li key={chip.key} className="max-w-full">
                <FileChip chip={chip} onRemove={() => chip.attachments.forEach((a) => removeAttachment(a.id))} />
              </li>
            ))}
            {region && (
              <li className="max-w-full">
                <RegionChip region={region} onRemove={() => setPendingRegion(null)} />
              </li>
            )}
          </ul>
        )}
        {note && <p className="text-[11.5px] text-muted">{note}</p>}
        <label className="sr-only" htmlFor="composer-input">
          Message
        </label>
        <textarea
          id="composer-input"
          ref={inputRef}
          rows={1}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={hasMessages ? 'Ask a follow-up…' : 'Describe an analysis… or drop GeoTIFFs here'}
          className="resize-none bg-transparent outline-none text-body text-text placeholder:text-faint"
        />
        {quota && (
          <p className="flex items-center gap-[7px] text-[11.5px] text-canc-ink" data-testid="quota-line">
            <span aria-hidden className="w-2 h-2 rounded-full bg-canc flex-none" />
            {quota}
          </p>
        )}
        {offline && (
          <p className="flex items-center gap-[7px] text-[11.5px] text-muted" data-testid="assistant-offline">
            <span aria-hidden className="w-2 h-2 rounded-full bg-canc flex-none" />
            {OFFLINE_LINE}
          </p>
        )}
        <div className="flex items-center gap-[6px]">
          <button
            type="button"
            onClick={onOpenFilePicker}
            aria-label="Attach GeoTIFF"
            title="Attach GeoTIFF"
            className="w-7 h-7 rounded-control border border-border text-[15px] text-muted hover:text-text hover:bg-surface2"
          >
            +
          </button>
          <button type="button" onClick={fromMap} className={control}>
            From map
          </button>
          <div ref={libraryRef} className="relative">
            <button
              type="button"
              onClick={() => setLibraryOpen((open) => !open)}
              aria-expanded={libraryOpen}
              className={control}
            >
              From library
            </button>
            {libraryOpen && <LibraryPicker onClose={closeLibrary} within={libraryRef} />}
          </div>
          <div className="flex-1" />
          <span className="text-[11px] text-muted">Enter to send · Shift+Enter newline</span>
          <button
            type="button"
            onClick={onSend}
            disabled={!canSend}
            aria-label="Send"
            className="w-[30px] h-[30px] rounded-[7px] bg-accent text-on-accent flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ArrowUp size={16} strokeWidth={2.25} aria-hidden />
          </button>
        </div>
      </div>
    </div>
  );
}
