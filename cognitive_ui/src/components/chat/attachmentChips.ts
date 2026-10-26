import type { MessageAttachment } from '../../types';

export interface AttachmentChip<T extends MessageAttachment = MessageAttachment> {
  key: string;
  name: string;
  line: string;
  attachments: T[];
}

/**
 * "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01": the metadata line with
 * the size before the date, which the server's line ends with.
 */
function fileLine(a: MessageAttachment): string {
  const { line = '', size_label: size, date } = a.metadata ?? {};
  if (!size) return line;
  const dateSuffix = date ? ` · ${date}` : '';
  if (dateSuffix && line.endsWith(dateSuffix)) return `${line.slice(0, -dateSuffix.length)} · ${size}${dateSuffix}`;
  return [line, size].filter(Boolean).join(' · ');
}

/**
 * The chips of the composer and of sent messages: one per group (the sample pair is
 * one chip with the group's name and line), one per other file with its metadata
 * line and size. Keyed by path: the two sample files share one id.
 */
export function attachmentChips<T extends MessageAttachment>(attachments: T[]): AttachmentChip<T>[] {
  const chips: AttachmentChip<T>[] = [];
  const byGroup = new Map<string, AttachmentChip<T>>();
  for (const a of attachments) {
    if (a.group) {
      const existing = byGroup.get(a.group.name);
      if (existing) {
        existing.attachments.push(a);
        continue;
      }
      const chip = { key: `group:${a.group.name}`, name: a.group.name, line: a.group.line, attachments: [a] };
      byGroup.set(a.group.name, chip);
      chips.push(chip);
      continue;
    }
    chips.push({ key: `file:${a.path ?? a.id}`, name: a.filename, line: fileLine(a), attachments: [a] });
  }
  return chips;
}
