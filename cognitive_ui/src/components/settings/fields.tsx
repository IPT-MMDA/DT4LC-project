import { useId, useState, type ReactNode } from 'react';
import { X } from 'lucide-react';
import { cn } from '../../utils/cn';
import { inputClass, smallButton, type SettingsForm } from './form';
import { sourceLabel, type Field } from './settingsModel';

export type Hue = 'done' | 'canc' | 'run' | 'muted' | 'fail';
const DOT: Record<Hue, string> = { done: 'bg-done', canc: 'bg-canc', run: 'bg-run', muted: 'bg-faint', fail: 'bg-fail' };
const LINE: Record<Hue, string> = {
  done: 'text-done-ink',
  canc: 'text-canc-ink',
  run: 'text-run-ink',
  muted: 'text-muted',
  fail: 'text-fail-ink',
};

/** A status as glyph + word + hue (DESIGN §2), as on the Capabilities page. */
export function Status({ hue, children, testId }: { hue: Hue; children: ReactNode; testId?: string }) {
  return (
    <span data-testid={testId} data-hue={hue} className={cn('inline-flex items-center gap-[5px] font-medium', LINE[hue])}>
      <span aria-hidden className={cn('w-[6px] h-[6px] rounded-full flex-none', DOT[hue])} />
      <span>{children}</span>
    </span>
  );
}

/** "from .env" / "override" / "default": where the saved value comes from. */
export function SourceTag({ field }: { field: Field }) {
  if (!field.source) return null;
  return (
    <span
      data-testid="setting-source"
      data-source={field.source}
      className={cn(
        'font-mono text-[10.5px] px-[6px] py-[1px] rounded-pill',
        field.source === 'override' ? 'bg-accent-tint text-accent-ink' : 'bg-surface2 text-muted',
      )}
    >
      {sourceLabel(field.source)}
    </span>
  );
}

/** Errors from a 422 for `path`, under its input. */
export function FieldErrors({ id, errors }: { id: string; errors: string[] | undefined }) {
  if (!errors?.length) return null;
  return (
    <div id={id} data-testid="field-error" className="flex flex-col gap-[2px]">
      {errors.map((msg) => (
        <p key={msg} className="text-[12px] text-fail-ink leading-[1.45]">
          {msg}
        </p>
      ))}
    </div>
  );
}

/**
 * One editable setting: label, source tag, "Reset to .env" (for an override), the input, and
 * the 422 errors. `children` gets the error element's id for `aria-describedby`, and whether a
 * reset is pending (the input is then disabled until Undo or Save).
 */
export function FieldRow({
  form,
  path,
  label,
  labelFor,
  hint,
  children,
}: {
  form: SettingsForm;
  path: string;
  label: string;
  labelFor?: string;
  hint?: string;
  children: (props: { describedBy: string | undefined; resetting: boolean }) => ReactNode;
}) {
  const errorId = useId();
  const field = form.field(path);
  const resetting = form.edits[path] === null;
  const errors = form.errors[path];
  const changed = form.edits[path] !== undefined && !resetting;
  return (
    <div data-testid="setting-field" data-path={path} className="flex flex-col gap-[6px]">
      <div className="flex items-center gap-2 flex-wrap">
        {labelFor ? (
          <label htmlFor={labelFor} className="text-control font-medium text-text">
            {label}
          </label>
        ) : (
          <span className="text-control font-medium text-text">{label}</span>
        )}
        <SourceTag field={field} />
        {changed && <span className="text-[11px] text-accent-ink">changed</span>}
        {field.source === 'override' && !resetting && (
          <button type="button" className="ml-auto text-[12px] text-muted hover:text-text underline" onClick={() => form.reset(path)}>
            Reset to .env
          </button>
        )}
      </div>
      {hint && <p className="text-[12px] text-muted">{hint}</p>}
      {children({ describedBy: errors?.length ? errorId : undefined, resetting })}
      {resetting && (
        <p className="text-[12px] text-run-ink flex items-center gap-2">
          Goes back to the .env value when you save.
          <button type="button" className="underline" onClick={() => form.undo(path)}>
            Undo
          </button>
        </p>
      )}
      <FieldErrors id={errorId} errors={errors} />
    </div>
  );
}

/**
 * An editable list of emails or domains as chips: × removes one, the input adds one or several
 * (Enter, or a comma/space-separated paste). Values are trimmed and lower-cased; the server validates.
 */
export function ChipList({
  values,
  onChange,
  label,
  placeholder,
  disabled,
  describedBy,
}: {
  values: string[];
  onChange: (values: string[]) => void;
  label: string;
  placeholder: string;
  disabled?: boolean;
  describedBy?: string;
}) {
  const [draft, setDraft] = useState('');
  const inputId = useId();
  const add = () => {
    const items = draft
      .split(/[\s,;]+/)
      .map((v) => v.trim().toLowerCase())
      .filter(Boolean);
    if (items.length) onChange([...values, ...items.filter((v, i) => !values.includes(v) && items.indexOf(v) === i)]);
    setDraft('');
  };
  return (
    <div className="flex flex-col gap-2">
      {values.length > 0 ? (
        <ul aria-label={label} className="flex flex-wrap gap-[6px]">
          {values.map((value) => (
            <li
              key={value}
              data-testid="chip"
              className="inline-flex items-center gap-1 pl-[9px] pr-[3px] py-[2px] rounded-pill bg-surface2 text-control text-text2"
            >
              <span className="font-mono text-[12px]">{value}</span>
              <button
                type="button"
                aria-label={`Remove ${value}`}
                disabled={disabled}
                onClick={() => onChange(values.filter((v) => v !== value))}
                className="w-5 h-5 rounded-pill flex items-center justify-center text-muted enabled:hover:bg-border2 enabled:hover:text-text"
              >
                <X aria-hidden className="w-3 h-3" />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[12px] text-muted">None.</p>
      )}
      <div className="flex gap-2">
        <label htmlFor={inputId} className="sr-only">
          {`Add to ${label}`}
        </label>
        <input
          id={inputId}
          type="text"
          value={draft}
          disabled={disabled}
          placeholder={placeholder}
          aria-describedby={describedBy}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              add();
            }
          }}
          className={cn(inputClass, 'flex-1 max-w-[360px]')}
        />
        <button type="button" className={smallButton} disabled={disabled || !draft.trim()} onClick={add}>
          Add
        </button>
      </div>
    </div>
  );
}

/** A read-only row of the Data sources and Server sections. */
export function InfoRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[180px_1fr] gap-[14px] px-4 py-[10px] border-b border-border2 last:border-b-0 items-baseline">
      <span className="font-medium">{label}</span>
      <div className="text-text2 min-w-0 flex flex-col gap-[2px]">{children}</div>
    </div>
  );
}
