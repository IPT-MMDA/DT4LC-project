import type { Attachment } from '../../types';

export interface AttachmentChip {
  key: string;
  name: string;
  line: string;
  attachments: Attachment[];
}

/**
 * The composer's chips: one per group (the sample pair is one chip with the
 * group's name and line), one per other file with `metadata.line · size_label`.
 */
export function attachmentChips(attachments: Attachment[]): AttachmentChip[] {
  const chips: AttachmentChip[] = [];
  const byGroup = new Map<string, AttachmentChip>();
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
    const line = [a.metadata?.line, a.metadata?.size_label].filter(Boolean).join(' · ');
    chips.push({ key: `file:${a.path}`, name: a.filename, line, attachments: [a] });
  }
  return chips;
}
