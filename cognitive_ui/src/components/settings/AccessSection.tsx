import { useId } from 'react';
import { cn } from '../../utils/cn';
import { ChipList, FieldRow, Status } from './fields';
import { card, inputClass, type SettingsForm } from './form';

const LISTS = [
  ['access.allowed_emails', 'Allowed emails', 'Signed-in users with these addresses are members.', 'name@example.org'],
  ['access.allowed_domains', 'Allowed domains', 'Any verified address at these domains is a member.', 'unige.ch'],
  ['access.operator_emails', 'Operator emails', 'Operators can open Settings. Your own address cannot be removed.', 'name@example.org'],
] as const;

const NUMBERS = [
  ['access.guest_analyses_per_hour', 'Guest analyses per hour', 'guest_analyses_per_hour'],
  ['access.guest_session_hours', 'Guest session hours', 'guest_session_hours'],
] as const;

function NumberField({ form, path, label, bound }: { form: SettingsForm; path: string; label: string; bound: string }) {
  const id = useId();
  const bounds = form.doc.access.bounds[bound];
  const value = form.value<number | string>(path);
  if (!form.field(path).editable) {
    return (
      <p className="text-control text-text2">
        {label}: <span className="font-mono">{value}</span>
      </p>
    );
  }
  return (
    <FieldRow form={form} path={path} label={label} labelFor={id} hint={bounds ? `${bounds.min}–${bounds.max}` : undefined}>
      {({ describedBy, resetting }) => (
        <input
          id={id}
          type="number"
          inputMode="numeric"
          min={bounds?.min}
          max={bounds?.max}
          step={1}
          value={value}
          disabled={resetting}
          aria-describedby={describedBy}
          // An empty or partial entry is sent as typed: the server's 422 says what is wrong.
          onChange={(e) => form.set(path, e.target.value === '' || Number.isNaN(e.target.valueAsNumber) ? e.target.value : e.target.valueAsNumber)}
          className={cn(inputClass, 'w-[120px] font-mono')}
        />
      )}
    </FieldRow>
  );
}

/** Access: the allowlists, the guest limits, and whether GitHub sign-in is set up. */
export function AccessSection({ form }: { form: SettingsForm }) {
  const { access } = form.doc;
  return (
    <div className="flex flex-col gap-5">
      <div className={cn(card, 'gap-1')}>
        <span data-testid="sign-in-status">
          {access.sign_in.configured ? (
            <Status hue="done">Sign-in configured</Status>
          ) : (
            <Status hue="muted">Sign-in not configured</Status>
          )}
        </span>
        <p className="text-[12px] text-muted">
          {access.sign_in.configured
            ? 'GitHub sign-in is on. The lists below decide who becomes a member or an operator.'
            : 'GitHub sign-in needs GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET and OAUTH_REDIRECT_BASE in the server’s environment. Until then everyone is a guest.'}
        </p>
      </div>

      <div className={cn(card, 'gap-5')}>
        {LISTS.map(([path, label, hint, placeholder]) =>
          form.field(path).editable ? (
            <FieldRow key={path} form={form} path={path} label={label} hint={hint}>
              {({ describedBy, resetting }) => (
                <ChipList
                  label={label}
                  placeholder={placeholder}
                  values={form.value<string[]>(path)}
                  disabled={resetting}
                  describedBy={describedBy}
                  onChange={(values) => form.set(path, values)}
                />
              )}
            </FieldRow>
          ) : (
            <div key={path} className="flex flex-col gap-1">
              <span className="text-control font-medium">{label}</span>
              <p className="font-mono text-[12px] text-text2">{form.value<string[]>(path).join(', ') || 'None.'}</p>
            </div>
          ),
        )}
      </div>

      <div className={cn(card, 'gap-5')}>
        {NUMBERS.map(([path, label, bound]) => (
          <NumberField key={path} form={form} path={path} label={label} bound={bound} />
        ))}
      </div>
    </div>
  );
}
