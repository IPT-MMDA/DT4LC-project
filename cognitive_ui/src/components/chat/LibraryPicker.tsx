import { useEffect, type RefObject } from 'react';
import { useListFiles, type GeoTIFFFile } from '../../api/hooks/useUpload';
import { useChatStore } from '../../store/useChatStore';
import type { Attachment } from '../../types';

function fileAttachment(f: GeoTIFFFile): Attachment {
  return {
    id: f.id,
    filename: f.filename,
    path: f.path,
    mime_type: 'image/tiff',
    size_bytes: f.size_bytes,
    metadata: f.metadata ?? undefined,
  };
}

/**
 * "From library": the caller's uploads (GET /v1/files) in a small popover; a click
 * attaches the file to the composer. The full list is the Library page (/app/library).
 */
export function LibraryPicker({ onClose, within }: { onClose: () => void; within: RefObject<HTMLElement> }) {
  const { data: files, isPending, isError } = useListFiles();
  const uploads = (files ?? []).filter((f) => f.source === 'upload');

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    const onPointer = (e: PointerEvent) => {
      // `within` holds the toggle button too, so a click on it toggles instead of reopening.
      if (within.current && !within.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointer);
    };
  }, [onClose, within]);

  return (
    <div
      role="dialog"
      aria-label="Your uploads"
      className="absolute bottom-full left-0 mb-2 w-[320px] max-h-[280px] overflow-y-auto bg-surface border border-border rounded-card p-2 z-20 shadow-composer"
    >
      <div className="px-2 pt-1 pb-2 text-label uppercase text-muted">Your uploads</div>
      {isPending && <p className="px-2 py-1 text-control text-muted">Loading…</p>}
      {isError && <p className="px-2 py-1 text-control text-fail-ink">Couldn't load your files.</p>}
      {!isPending && !isError && uploads.length === 0 && (
        <p className="px-2 py-1 text-control text-muted">No uploads yet. Use + to upload a GeoTIFF.</p>
      )}
      <ul className="flex flex-col">
        {uploads.map((f) => (
          <li key={f.path}>
            <button
              type="button"
              onClick={() => {
                useChatStore.getState().addAttachment(fileAttachment(f));
                onClose();
              }}
              className="w-full text-left px-2 py-[6px] rounded-control hover:bg-surface2"
            >
              <div className="text-[12px] font-mono truncate">{f.filename}</div>
              {f.metadata?.line && <div className="text-[11px] text-muted truncate">{f.metadata.line}</div>}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
